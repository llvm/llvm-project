// RUN: mlir-opt %s | FileCheck %s

// Verify round-trip parsing and printing of the builtin quantile type.

// CHECK: func private @quantile_ui4_f16(quantile<ui4:f16, {-1.000000e+00,0.000000e+00,1.000000e+00}>)
func.func private @quantile_ui4_f16(quantile<ui4:f16, {-1.0,0.0,1.0}>) -> ()

// CHECK: func private @quantile_si8_f32(quantile<si8:f32, {-1.000000e+00,-5.000000e-01,0.000000e+00,5.000000e-01,1.000000e+00}>)
func.func private @quantile_si8_f32(quantile<si8:f32, {-1.0,-0.5,0.0,0.5,1.0}>) -> ()

// CHECK: func private @quantile_i8_f32(quantile<i8:f32, {-1.000000e+00,0.000000e+00,1.000000e+00}>)
func.func private @quantile_i8_f32(quantile<i8:f32, {-1.0,0.0,1.0}>) -> ()

// CHECK: func private @quantile_f8_f32(quantile<f8E4M3FN:f32, {-1.000000e+00,0.000000e+00,1.000000e+00}>)
func.func private @quantile_f8_f32(quantile<f8E4M3FN:f32, {-1.0,0.0,1.0}>) -> ()

// CHECK: func private @quantile_ui4_bf16(quantile<ui4:bf16, {-1.000000e+00,0.000000e+00,1.000000e+00}>)
func.func private @quantile_ui4_bf16(quantile<ui4:bf16, {-1.0,0.0,1.0}>) -> ()

// CHECK: func private @quantile_as_return() -> quantile<ui4:f16, {-1.000000e+00,0.000000e+00,1.000000e+00}>
func.func private @quantile_as_return() -> quantile<ui4:f16, {-1.0,0.0,1.0}>

// Verify use as memref element type (requires MemRefElementTypeInterface).
// CHECK: func private @quantile_in_memref(memref<8xquantile<ui4:f16, {-1.000000e+00,0.000000e+00,1.000000e+00}>>)
func.func private @quantile_in_memref(memref<8xquantile<ui4:f16, {-1.0,0.0,1.0}>>) -> ()

// Verify use as tensor element type.
// CHECK: func private @quantile_in_tensor(tensor<16xquantile<ui4:f16, {-1.000000e+00,0.000000e+00,1.000000e+00}>>)
func.func private @quantile_in_tensor(tensor<16xquantile<ui4:f16, {-1.0,0.0,1.0}>>) -> ()

// Verify use in multidimensional tensors.
// CHECK: func private @quantile_in_ranked_tensor(tensor<16x16x1x1xquantile<ui4:f16, {-1.000000e+00,0.000000e+00,1.000000e+00}>>)
func.func private @quantile_in_ranked_tensor(tensor<16x16x1x1xquantile<ui4:f16, {-1.0,0.0,1.0}>>) -> ()

// Verify use in unranked tensors.
// CHECK: func private @quantile_in_unranked_tensor(tensor<*xquantile<ui4:f16, {-1.000000e+00,0.000000e+00,1.000000e+00}>>)
func.func private @quantile_in_unranked_tensor(tensor<*xquantile<ui4:f16, {-1.0,0.0,1.0}>>) -> ()

// Verify NF4-style 16-entry quantile table
// CHECK-LABEL: @nf4_16_values
// CHECK-SAME: quantile<ui4:f16, {
func.func private @nf4_16_values(quantile<ui4:f16, {
  -1.0,-0.6961928009986877,-0.5250730514526367,-0.39491748809814453,
  -0.28444138169288635,-0.18477343022823334,-0.09105003625154495,0.0,
  0.07958029955625534,0.16093020141124725,0.24611230194568634,
  0.33791524171829224,0.44070982933044434,0.5626170039176941,
  0.7229568362236023,1.0}>) -> ()

// Verify explicit storage min/max range (unsigned storage, narrowed range).
// CHECK: func private @quantile_with_range(quantile<ui4:f16, {-1.000000e+00,0.000000e+00,1.000000e+00}><0:7>)
func.func private @quantile_with_range(quantile<ui4:f16, {-1.0,0.0,1.0}><0:7>) -> ()

// Verify explicit range is preserved through round-trip.
// CHECK: func private @quantile_signed_range(quantile<si8:f32, {-1.000000e+00,0.000000e+00,1.000000e+00}><-100:100>)
func.func private @quantile_signed_range(quantile<si8:f32, {-1.0,0.0,1.0}><-100:100>) -> ()

// Verify negative-only quantile list.
// CHECK: func private @quantile_negatives(quantile<si4:f32, {-1.000000e+00,-5.000000e-01,-2.500000e-01}>)
func.func private @quantile_negatives(quantile<si4:f32, {-1.0,-0.5,-0.25}>) -> ()

// Verify that a single quantile value is accepted.
// CHECK: func private @quantile_single_value(quantile<si8:f32, {1.000000e+00}>)
func.func private @quantile_single_value(quantile<si8:f32, {1.0}>) -> ()
