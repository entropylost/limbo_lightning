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

// https://nullprogram.com/blog/2018/07/31/
#[tracked]
fn hash(x: Expr<u32>) -> Expr<u32> {
    let x = x.var();
    *x ^= x >> 17;
    *x *= 0xed5ad4bb;
    *x ^= x >> 11;
    *x *= 0xac4c1b51;
    *x ^= x >> 15;
    *x *= 0x31848bab;
    *x ^= x >> 14;
    **x
}

#[tracked]
fn rand(pos: Expr<Vec2<u32>>, t: Expr<u32>, c: u32) -> Expr<u32> {
    let input = t + pos.x * GRID_SIZE + pos.y * GRID_SIZE * GRID_SIZE + c * 1063; //* GRID_SIZE * GRID_SIZE * GRID_SIZE;
    hash(input)
}

#[tracked]
fn rand_f32(pos: Expr<Vec2<u32>>, t: Expr<u32>, c: u32) -> Expr<f32> {
    rand(pos, t, c).as_f32() / u32::MAX as f32
}

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

    let aq: VField<f32, Vec2<i32>> = fields.create_bind(
        "air-quality",
        domain.map_texture(device.create_tex2d(PixelStorage::Float1, GRID_SIZE, GRID_SIZE, 1)),
    );

    let finder: VField<Vec2<i32>, Vec2<i32>> = fields.create_bind(
        "finder",
        domain.map_texture(device.create_tex2d(PixelStorage::Int2, GRID_SIZE, GRID_SIZE, 1)),
    );
    let valid: VField<bool, Vec2<i32>> = *fields.create_bind(
        "valid",
        domain.map_buffer_morton(device.create_buffer(GRID_SIZE as usize * GRID_SIZE as usize)),
    );
    let dist: VField<f32, Vec2<i32>> = fields.create_bind(
        "dist",
        domain.map_texture(device.create_tex2d(PixelStorage::Float1, GRID_SIZE, GRID_SIZE, 1)),
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
                Vec3::splat_expr(0.0_f32)
            } else {
                if el.expr(&charge) != 0 {
                    Vec3::expr(1.0, 0.9, 0.2)
                } else {
                    Vec3::splat(0.9) * el.expr(&aq)
                }
                // let c =
            };
            *display_el.var(&display) = color.extend(1.0);
        }),
    );

    let init_aq = Kernel::<fn()>::build(
        &device,
        &domain,
        track!(&|mut el| {
            *el.var(&aq) = rand_f32(el.cast_u32(), 0.expr(), 0) * 0.1
                + rand_f32((*el / 2_i32).cast_u32(), 0.expr(), 1) * 0.1
                + rand_f32((*el / 4_i32).cast_u32(), 0.expr(), 4) * 0.1
                + 1.0;
        }),
    );

    let init_finders = Kernel::<fn()>::build(
        &device,
        &domain,
        track!(&|mut el| {
            *el.var(&dist) = f32::MAX;
            *el.var(&finder) = *el;
        }),
    );

    let propegate_nearest = Kernel::<fn()>::build(
        &device,
        &domain,
        track!(&|mut el| {
            let is_valid = el.expr(&valid).var();
            let best_dist = el.expr(&dist).var();
            let best_finder = el.expr(&finder).var();
            if el.expr(&ground) {
                *best_dist = 0.0;
                *best_finder = *el;
                *is_valid = true;
            }
            domain.on_adjacent(&el, |mut el| {
                let this_valid = el.expr(&valid);
                let dist = (el.expr(&dist) + 1.0) * el.expr(&aq);
                if this_valid && (!is_valid || dist < best_dist) {
                    *best_dist = dist;
                    *best_finder = *el;
                    *is_valid = true;
                }
            });
            *el.var(&dist) = best_dist;
            *el.var(&finder) = best_finder;
            *el.var(&valid) = is_valid;
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
                el.atomic(&next_charge).fetch_min(0);
                return;
            }
            let finder = el.expr(&finder);
            if (finder != pos).any() {
                // safety
                let mut fel = domain.index(finder, &el);
                if fel.expr(&charge) < MAX_CHARGE {
                    let fill = 1; // luisa::min(MAX_CHARGE - fel.expr(&charge), el.expr(&charge));
                    fel.atomic(&next_charge).fetch_add(fill);
                    el.atomic(&next_charge).fetch_sub(fill);
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
    graph.add((init_finders.dispatch(), init_aq.dispatch()).chain());
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
