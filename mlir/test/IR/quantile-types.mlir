// RUN: mlir-opt %s | FileCheck %s

// Verify round-trip parsing and printing of the builtin quantile type.

// CHECK-LABEL: func private @quantile_ui4_f16
// CHECK-SAME: quantile<ui4:f16, {-1.000000e+00,-8.667000e-01,-7.333000e-01,-6.000000e-01,-4.667000e-01,-3.333000e-01,-2.000000e-01,-6.670000e-02,6.670000e-02,2.000000e-01,3.333000e-01,4.667000e-01,6.000000e-01,7.333000e-01,8.667000e-01,1.000000e+00}>
func.func private @quantile_ui4_f16(quantile<ui4:f16, {-1.0,-0.8667,-0.7333,-0.6,-0.4667,-0.3333,-0.2,-0.0667,0.0667,0.2,0.3333,0.4667,0.6,0.7333,0.8667,1.0}>) -> ()

// CHECK: func private @quantile_si8_f32(quantile<si8:f32, {-1.000000e+00,-5.000000e-01,0.000000e+00,5.000000e-01,1.000000e+00}><-2:2>)
func.func private @quantile_si8_f32(quantile<si8:f32, {-1.0,-0.5,0.0,0.5,1.0}><-2:2>) -> ()

// CHECK: func private @quantile_i8_f32(quantile<i8:f32, {-1.000000e+00,0.000000e+00,1.000000e+00}><-1:1>)
func.func private @quantile_i8_f32(quantile<i8:f32, {-1.0,0.0,1.0}><-1:1>) -> ()

// CHECK: func private @quantile_f8_f32(quantile<f8E4M3FN:f32, {-1.000000e+00,0.000000e+00,1.000000e+00}><-1:1>)
func.func private @quantile_f8_f32(quantile<f8E4M3FN:f32, {-1.0,0.0,1.0}><-1:1>) -> ()

// CHECK-LABEL: func private @quantile_ui4_bf16
// CHECK-SAME: quantile<ui4:bf16, {-1.000000e+00,-8.667000e-01,-7.333000e-01,-6.000000e-01,-4.667000e-01,-3.333000e-01,-2.000000e-01,-6.670000e-02,6.670000e-02,2.000000e-01,3.333000e-01,4.667000e-01,6.000000e-01,7.333000e-01,8.667000e-01,1.000000e+00}>
func.func private @quantile_ui4_bf16(quantile<ui4:bf16, {-1.0,-0.8667,-0.7333,-0.6,-0.4667,-0.3333,-0.2,-0.0667,0.0667,0.2,0.3333,0.4667,0.6,0.7333,0.8667,1.0}>) -> ()

// CHECK-LABEL: func private @quantile_as_return
// CHECK-SAME: quantile<ui4:f16, {-1.000000e+00,-8.667000e-01,-7.333000e-01,-6.000000e-01,-4.667000e-01,-3.333000e-01,-2.000000e-01,-6.670000e-02,6.670000e-02,2.000000e-01,3.333000e-01,4.667000e-01,6.000000e-01,7.333000e-01,8.667000e-01,1.000000e+00}>
func.func private @quantile_as_return() -> quantile<ui4:f16, {-1.0,-0.8667,-0.7333,-0.6,-0.4667,-0.3333,-0.2,-0.0667,0.0667,0.2,0.3333,0.4667,0.6,0.7333,0.8667,1.0}>

// Verify use as memref element type (requires MemRefElementTypeInterface).
// CHECK-LABEL: func private @quantile_in_memref
// CHECK-SAME: memref<8xquantile<ui4:f16, {-1.000000e+00,-8.667000e-01,-7.333000e-01,-6.000000e-01,-4.667000e-01,-3.333000e-01,-2.000000e-01,-6.670000e-02,6.670000e-02,2.000000e-01,3.333000e-01,4.667000e-01,6.000000e-01,7.333000e-01,8.667000e-01,1.000000e+00}>>
func.func private @quantile_in_memref(memref<8xquantile<ui4:f16, {-1.0,-0.8667,-0.7333,-0.6,-0.4667,-0.3333,-0.2,-0.0667,0.0667,0.2,0.3333,0.4667,0.6,0.7333,0.8667,1.0}>>) -> ()

// Verify use as tensor element type.
// CHECK-LABEL: func private @quantile_in_tensor
// CHECK-SAME: tensor<16xquantile<ui4:f16, {-1.000000e+00,-8.667000e-01,-7.333000e-01,-6.000000e-01,-4.667000e-01,-3.333000e-01,-2.000000e-01,-6.670000e-02,6.670000e-02,2.000000e-01,3.333000e-01,4.667000e-01,6.000000e-01,7.333000e-01,8.667000e-01,1.000000e+00}>>
func.func private @quantile_in_tensor(tensor<16xquantile<ui4:f16, {-1.0,-0.8667,-0.7333,-0.6,-0.4667,-0.3333,-0.2,-0.0667,0.0667,0.2,0.3333,0.4667,0.6,0.7333,0.8667,1.0}>>) -> ()

// Verify use in multidimensional tensors.
// CHECK-LABEL: func private @quantile_in_ranked_tensor
// CHECK-SAME: tensor<16x16x1x1xquantile<ui4:f16, {-1.000000e+00,-8.667000e-01,-7.333000e-01,-6.000000e-01,-4.667000e-01,-3.333000e-01,-2.000000e-01,-6.670000e-02,6.670000e-02,2.000000e-01,3.333000e-01,4.667000e-01,6.000000e-01,7.333000e-01,8.667000e-01,1.000000e+00}>>
func.func private @quantile_in_ranked_tensor(tensor<16x16x1x1xquantile<ui4:f16, {-1.0,-0.8667,-0.7333,-0.6,-0.4667,-0.3333,-0.2,-0.0667,0.0667,0.2,0.3333,0.4667,0.6,0.7333,0.8667,1.0}>>) -> ()

// Verify use in unranked tensors.
// CHECK-LABEL: func private @quantile_in_unranked_tensor
// CHECK-SAME: tensor<*xquantile<ui4:f16, {-1.000000e+00,-8.667000e-01,-7.333000e-01,-6.000000e-01,-4.667000e-01,-3.333000e-01,-2.000000e-01,-6.670000e-02,6.670000e-02,2.000000e-01,3.333000e-01,4.667000e-01,6.000000e-01,7.333000e-01,8.667000e-01,1.000000e+00}>>
func.func private @quantile_in_unranked_tensor(tensor<*xquantile<ui4:f16, {-1.0,-0.8667,-0.7333,-0.6,-0.4667,-0.3333,-0.2,-0.0667,0.0667,0.2,0.3333,0.4667,0.6,0.7333,0.8667,1.0}>>) -> ()

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
// CHECK: func private @quantile_with_range(quantile<ui4:f16, {-1.000000e+00,-7.500000e-01,-5.000000e-01,-2.500000e-01,0.000000e+00,2.500000e-01,5.000000e-01,1.000000e+00}><0:7>)
func.func private @quantile_with_range(quantile<ui4:f16, {-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,1.0}><0:7>) -> ()

// Verify explicit range is preserved through round-trip.
// CHECK: func private @quantile_signed_range(quantile<si8:f32, {-1.000000e+00,0.000000e+00,1.000000e+00}><-1:1>)
func.func private @quantile_signed_range(quantile<si8:f32, {-1.0,0.0,1.0}><-1:1>) -> ()

// Verify signed 4-bit storage type uses full 16-entry LUT (all-negative values).
// CHECK-LABEL: func private @quantile_negatives
// CHECK-SAME: quantile<si4:f32, {-2.000000e+00,-1.875000e+00,-1.750000e+00,-1.625000e+00,-1.500000e+00,-1.375000e+00,-1.250000e+00,-1.125000e+00,-1.000000e+00,-8.750000e-01,-7.500000e-01,-6.250000e-01,-5.000000e-01,-3.750000e-01,-2.500000e-01,-1.250000e-01}>
func.func private @quantile_negatives(quantile<si4:f32, {-2.0,-1.875,-1.75,-1.625,-1.5,-1.375,-1.25,-1.125,-1.0,-0.875,-0.75,-0.625,-0.5,-0.375,-0.25,-0.125}>) -> ()

// Verify minimal 2-entry LUT for 1-bit unsigned storage type.
// CHECK: func private @quantile_ui1_f16(quantile<ui1:f16, {-1.000000e+00,1.000000e+00}>)
func.func private @quantile_ui1_f16(quantile<ui1:f16, {-1.0,1.0}>) -> ()

// Verify LUT values in descending order
// Storage is ui4 with explicit <0:7> range (8 entries).
// CHECK: func private @quantile_descending(quantile<ui4:f16, {1.000000e+00,7.500000e-01,5.000000e-01,2.500000e-01,0.000000e+00,-2.500000e-01,-5.000000e-01,-1.000000e+00}><0:7>)
func.func private @quantile_descending(quantile<ui4:f16, {1.0,0.75,0.5,0.25,0.0,-0.25,-0.5,-1.0}><0:7>) -> ()

// Verify LUT values in an arbitrary order
// Storage is ui4 with explicit <0:7> range (8 entries).
// CHECK: func private @quantile_random_order(quantile<ui4:f16, {0.000000e+00,-5.000000e-01,1.000000e+00,-2.500000e-01,7.500000e-01,-1.000000e+00,5.000000e-01,2.500000e-01}><0:7>)
func.func private @quantile_random_order(quantile<ui4:f16, {0.0,-0.5,1.0,-0.25,0.75,-1.0,0.5,0.25}><0:7>) -> ()
