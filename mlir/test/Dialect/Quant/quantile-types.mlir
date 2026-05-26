// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s

// -----
// Quantile type: ui4 storage with f16 expressed, 16 entries (default range 0..15).
// CHECK-LABEL: func private @quantile_ui4_f16
// CHECK-SAME: !quant.quantile<ui4:f16, {
func.func private @quantile_ui4_f16(!quant.quantile<ui4:f16, {-1.0,-0.8667,-0.7333,-0.6,-0.4667,-0.3333,-0.2,-0.0667,0.0667,0.2,0.3333,0.4667,0.6,0.7333,0.8667,1.0}>) -> ()

// -----
// Quantile type: si8 storage with f32 expressed, explicit range -2:2 (5 entries).
// CHECK: func private @quantile_si8_f32(!quant.quantile<si8:f32, {-1.000000e+00,-5.000000e-01,0.000000e+00,5.000000e-01,1.000000e+00}, <-2:2>>)
func.func private @quantile_si8_f32(!quant.quantile<si8:f32, {-1.0,-0.5,0.0,0.5,1.0}, <-2:2>>) -> ()

// -----
// Quantile type: i8 (signless) storage with f32 expressed, explicit range -1:1 (3 entries).
// CHECK: func private @quantile_i8_f32(!quant.quantile<i8:f32, {-1.000000e+00,0.000000e+00,1.000000e+00}, <-1:1>>)
func.func private @quantile_i8_f32(!quant.quantile<i8:f32, {-1.0,0.0,1.0}, <-1:1>>) -> ()

// -----
// Quantile type: f8E4M3FN float storage with f32 expressed, explicit range -1:1 (3 entries).
// CHECK: func private @quantile_f8_f32(!quant.quantile<f8E4M3FN:f32, {-1.000000e+00,0.000000e+00,1.000000e+00}, <-1:1>>)
func.func private @quantile_f8_f32(!quant.quantile<f8E4M3FN:f32, {-1.0,0.0,1.0}, <-1:1>>) -> ()

// -----
// Quantile type: ui4 storage with bf16 expressed, 16 entries.
// CHECK-LABEL: func private @quantile_ui4_bf16
// CHECK-SAME: !quant.quantile<ui4:bf16, {
func.func private @quantile_ui4_bf16(!quant.quantile<ui4:bf16, {-1.0,-0.8667,-0.7333,-0.6,-0.4667,-0.3333,-0.2,-0.0667,0.0667,0.2,0.3333,0.4667,0.6,0.7333,0.8667,1.0}>) -> ()

// -----
// Quantile type used as a return type.
// CHECK-LABEL: func private @quantile_as_return
// CHECK-SAME: !quant.quantile<ui4:f16, {
func.func private @quantile_as_return() -> !quant.quantile<ui4:f16, {-1.0,-0.8667,-0.7333,-0.6,-0.4667,-0.3333,-0.2,-0.0667,0.0667,0.2,0.3333,0.4667,0.6,0.7333,0.8667,1.0}>

// -----
// NF4-style 16-entry quantile table.
// CHECK-LABEL: @nf4_16_values
// CHECK-SAME: !quant.quantile<ui4:f16, {
func.func private @nf4_16_values(!quant.quantile<ui4:f16, {
  -1.0,-0.6961928009986877,-0.5250730514526367,-0.39491748809814453,
  -0.28444138169288635,-0.18477343022823334,-0.09105003625154495,0.0,
  0.07958029955625534,0.16093020141124725,0.24611230194568634,
  0.33791524171829224,0.44070982933044434,0.5626170039176941,
  0.7229568362236023,1.0}>) -> ()

// -----
// Explicit storage min/max range (unsigned storage, narrowed range 0..7, 8 entries).
// CHECK: func private @quantile_with_range(!quant.quantile<ui4:f16, {-1.000000e+00,-7.500000e-01,-5.000000e-01,-2.500000e-01,0.000000e+00,2.500000e-01,5.000000e-01,1.000000e+00}, <0:7>>)
func.func private @quantile_with_range(!quant.quantile<ui4:f16, {-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,1.0}, <0:7>>) -> ()

// -----
// Explicit range is preserved through round-trip.
// CHECK: func private @quantile_signed_range(!quant.quantile<si8:f32, {-1.000000e+00,0.000000e+00,1.000000e+00}, <-1:1>>)
func.func private @quantile_signed_range(!quant.quantile<si8:f32, {-1.0,0.0,1.0}, <-1:1>>) -> ()

// -----
// Signed 4-bit storage uses full 16-entry LUT (range -8..7).
// CHECK-LABEL: func private @quantile_negatives
// CHECK-SAME: !quant.quantile<si4:f32, {-2.000000e+00,-1.875000e+00,-1.750000e+00,-1.625000e+00,-1.500000e+00,-1.375000e+00,-1.250000e+00,-1.125000e+00,-1.000000e+00,-8.750000e-01,-7.500000e-01,-6.250000e-01,-5.000000e-01,-3.750000e-01,-2.500000e-01,-1.250000e-01}>
func.func private @quantile_negatives(!quant.quantile<si4:f32, {-2.0,-1.875,-1.75,-1.625,-1.5,-1.375,-1.25,-1.125,-1.0,-0.875,-0.75,-0.625,-0.5,-0.375,-0.25,-0.125}>) -> ()

// -----
// 1-bit unsigned storage: minimal 2-entry LUT.
// CHECK: func private @quantile_ui1_f16(!quant.quantile<ui1:f16, {-1.000000e+00,1.000000e+00}>)
func.func private @quantile_ui1_f16(!quant.quantile<ui1:f16, {-1.0,1.0}>) -> ()

// -----
// LUT values in descending order (ui4, explicit range 0:7, 8 entries).
// CHECK: func private @quantile_descending(!quant.quantile<ui4:f16, {1.000000e+00,7.500000e-01,5.000000e-01,2.500000e-01,0.000000e+00,-2.500000e-01,-5.000000e-01,-1.000000e+00}, <0:7>>)
func.func private @quantile_descending(!quant.quantile<ui4:f16, {1.0,0.75,0.5,0.25,0.0,-0.25,-0.5,-1.0}, <0:7>>) -> ()

// -----
// LUT values in arbitrary order (ui4, explicit range 0:7, 8 entries).
// CHECK: func private @quantile_random_order(!quant.quantile<ui4:f16, {0.000000e+00,-5.000000e-01,1.000000e+00,-2.500000e-01,7.500000e-01,-1.000000e+00,5.000000e-01,2.500000e-01}, <0:7>>)
func.func private @quantile_random_order(!quant.quantile<ui4:f16, {0.0,-0.5,1.0,-0.25,0.75,-1.0,0.5,0.25}, <0:7>>) -> ()