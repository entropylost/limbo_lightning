use std::{
    collections::HashSet,
    env::current_exe,
    f32::consts::{PI, TAU},
    time::{Duration, Instant},
};

use luisa::lang::{
    functions::{block_id, sync_block},
    types::{
        shared::Shared,
        vector::{Vec2, Vec3, Vec4},
    },
};
use sefirot::{graph::ComputeGraph, mapping::buffer::StaticDomain, prelude::*};
use sefirot_grid::GridDomain;
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
};

const GRID_SIZE: u32 = 256;
const SCALING: u32 = 8;
const MAX_CHARGE: u32 = 16;

#[derive(Debug, Clone, Copy)]
struct Runtime {
    cursor_pos: PhysicalPosition<f64>,
    t: u32,
}
impl Default for Runtime {
    fn default() -> Self {
        Self {
            cursor_pos: PhysicalPosition::new(0.0, 0.0),
            t: 0,
        }
    }
}

fn main() {
    let _ = color_eyre::install();
    luisa::init_logger();
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device("cuda");

    let event_loop = EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(
            GRID_SIZE * SCALING,
            GRID_SIZE * SCALING,
        ))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    let swapchain = device.create_swapchain(
        &window,
        &device.default_stream(),
        GRID_SIZE * SCALING,
        GRID_SIZE * SCALING,
        false,
        false,
        3,
    );
    let display_texture = device.create_tex2d::<Vec4<f32>>(
        swapchain.pixel_storage(),
        GRID_SIZE * SCALING,
        GRID_SIZE * SCALING,
        1,
    );
    let mut fields = FieldSet::new();
    let display_domain = StaticDomain::<2>::new(GRID_SIZE * SCALING, GRID_SIZE * SCALING);
    let display: VField<Vec4<f32>, Vec2<u32>> =
        fields.create_bind("display", display_domain.map_tex2d(display_texture.view(0)));

    let domain = GridDomain::new([0, 0], [GRID_SIZE; 2]);

    let nearest_ground: VField<Vec2<i32>, Vec2<i32>> = fields.create_bind(
        "nearest-ground",
        domain.map_texture(device.create_tex2d(PixelStorage::Int2, GRID_SIZE, GRID_SIZE, 1)),
    );
    let valid: VField<bool, Vec2<i32>> = *fields.create_bind(
        "ground",
        domain.map_buffer_morton(device.create_buffer(GRID_SIZE as usize * GRID_SIZE as usize)),
    );
    let nearest_ground_finder: VField<Vec2<i32>, Vec2<i32>> = fields.create_bind(
        "nearest-ground",
        domain.map_texture(device.create_tex2d(PixelStorage::Int2, GRID_SIZE, GRID_SIZE, 1)),
    );
    let charge: AField<u32, Vec2<i32>> = fields.create_bind(
        "charge",
        domain.map_buffer_morton(device.create_buffer(GRID_SIZE as usize * GRID_SIZE as usize)),
    );
    let next_charge: AField<u32, Vec2<i32>> = fields.create_bind(
        "next_charge",
        domain.map_buffer_morton(device.create_buffer(GRID_SIZE as usize * GRID_SIZE as usize)),
    );

    let ground: VField<bool, Vec2<i32>> = *fields.create_bind(
        "ground",
        domain.map_buffer_morton(device.create_buffer(GRID_SIZE as usize * GRID_SIZE as usize)),
    );

    let draw_kernel = Kernel::<fn()>::build(
        &device,
        &display_domain,
        track!(&|mut display_el| {
            let pos = (*display_el / SCALING).cast_i32();
            let mut el = domain.index(pos, &display_el);
            let color = if el.expr(&ground) {
                Vec3::splat_expr(1.0_f32)
            } else {
                if el.expr(&charge) != 0 {
                    Vec3::expr(0.5, 0.5, 0.0)
                } else if el.expr(&valid) {
                    Vec3::expr(0.0, 0.0, 0.2)
                } else {
                    Vec3::expr(0.0, 0.0, 0.0)
                }
                // let c = el.expr(&charge).cast_f32() / MAX_CHARGE as f32;
                // Vec3::expr(1.0, 0.9, 0.2) * c
            };
            *display_el.var(&display) = color.extend(1.0);
        }),
    );

    let update_valid = Kernel::<fn()>::build(
        &device,
        &domain,
        track!(&|mut el| {
            *el.var(&valid) = domain.index(el.expr(&nearest_ground), &el).expr(&ground);
        }),
    );

    let init_nearest_ground = Kernel::<fn()>::build(
        &device,
        &domain,
        track!(&|mut el| {
            *el.var(&nearest_ground) = *el;
            *el.var(&nearest_ground_finder) = *el;
        }),
    );

    let propegate_nearest = Kernel::<fn()>::build(
        &device,
        &domain,
        track!(&|mut el| {
            let best_dist = i32::MAX.var();
            let best_ground = (*el).var();
            let old_ground_finder = el.expr(&nearest_ground_finder);
            let best_ground_finder = (*el).var();
            let pos = *el;
            domain.on_adjacent(&el, |mut el| {
                let ground = el.expr(&nearest_ground);
                let valid = el.expr(&valid);
                if valid {
                    let delta = ground - *el;
                    let dist = delta.x * delta.x + delta.y * delta.y + 1;
                    if dist < best_dist {
                        *best_dist = dist;
                        *best_ground = ground;
                        *best_ground_finder = *el;
                    }
                }
            });
            let ground = el.expr(&nearest_ground);
            let valid = el.expr(&valid);
            if valid {
                let delta = ground - pos;
                let dist = delta.x * delta.x + delta.y * delta.y;
                if dist < best_dist {
                    *best_dist = dist;
                    *best_ground = ground;
                    *best_ground_finder = old_ground_finder;
                }
            }
            // if el.expr(&value) < IMF_CAP / 2 {
            //     *best_dist = 0;
            //     *best_ground = pos;
            // }
            // TODO: Also check the current out to see if it's also good?
            if best_dist == i32::MAX {
                return;
            }
            *el.var(&nearest_ground) = best_ground;
            *el.var(&nearest_ground_finder) = best_ground_finder;
        }),
    );

    let discharge = Kernel::<fn()>::build(
        &device,
        &domain,
        track!(&|mut el| {
            let pos = *el;
            if el.expr(&charge) == 0 {
                return;
            }
            if !el.expr(&valid) {
                return;
            }
            if el.expr(&ground) {
                el.atomic(&next_charge).fetch_sub(1);
                return;
            }
            let finder = el.expr(&nearest_ground_finder);
            if (finder != pos).any() {
                // safety
                let mut fel = domain.index(finder, &el);
                if fel.expr(&charge) < MAX_CHARGE {
                    fel.atomic(&next_charge).fetch_add(1);
                    el.atomic(&next_charge).fetch_sub(1);
                }
            }
        }),
    );
    let copy_charge = Kernel::<fn()>::build(
        &device,
        &domain,
        track!(&|mut el| {
            *el.var(&charge) = el.expr(&next_charge);
        }),
    );

    let write_wall = Kernel::<fn(Vec2<i32>)>::build(
        &device,
        &domain,
        track!(&|mut el, pos| {
            if (*el != pos).any() {
                return;
            }
            *el.var(&ground) = true;
        }),
    );
    let write_charge = Kernel::<fn(Vec2<i32>, u32)>::build(
        &device,
        &domain,
        track!(&|mut el, pos, c| {
            if (*el != pos).any() {
                return;
            }
            *el.var(&charge) = c;
            *el.var(&next_charge) = c;
        }),
    );

    let mut graph = ComputeGraph::new(&device);
    graph.add((init_nearest_ground.dispatch(), update_valid.dispatch()).chain());
    graph.execute_clear();

    let mut active_buttons = HashSet::new();

    let mut update_cursor = |active_buttons: &HashSet<MouseButton>, rt: &mut Runtime| {
        let pos = Vec2::new(
            (rt.cursor_pos.x as i32) / SCALING as i32,
            (rt.cursor_pos.y as i32) / SCALING as i32,
        );
        if active_buttons.contains(&MouseButton::Left) {
            write_wall.dispatch_blocking(&pos);
        }
        if active_buttons.contains(&MouseButton::Right) {
            write_charge.dispatch_blocking(&pos, &MAX_CHARGE);
        }
    };
    let update_cursor = &mut update_cursor;

    let mut rt = Runtime::default();

    let start = Instant::now();

    let dt = Duration::from_secs_f64(1.0 / 60.0);

    let mut avg_iter_time = 0.0;

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run(move |event, elwt| match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => {
                    elwt.exit();
                }
                WindowEvent::RedrawRequested => {
                    let scope = device.default_stream().scope();
                    scope.present(&swapchain, &display_texture);

                    let iter_st = Instant::now();
                    rt.t += 1;
                    update_cursor(&active_buttons, &mut rt);

                    graph.add(
                        (
                            propegate_nearest.dispatch(),
                            update_valid.dispatch(),
                            discharge.dispatch(),
                            copy_charge.dispatch(),
                            draw_kernel.dispatch(),
                        )
                            .chain(),
                    );
                    graph.execute_clear();

                    window.request_redraw();
                }
                WindowEvent::CursorMoved { position, .. } => {
                    rt.cursor_pos = position;
                    update_cursor(&active_buttons, &mut rt);
                }
                WindowEvent::MouseInput { button, state, .. } => {
                    match state {
                        ElementState::Pressed => {
                            active_buttons.insert(button);
                        }
                        ElementState::Released => {
                            active_buttons.remove(&button);
                        }
                    }
                    update_cursor(&active_buttons, &mut rt);
                }
                _ => (),
            },
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => (),
        })
        .unwrap();
}
