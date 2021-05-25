// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func @memref_cast(
func @memref_cast(%a: index, %b: index) -> memref<?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %c16 = constant 16 : index
  %1 = memref.alloc (%b) : memref<?xi8>
  %2 = memref.view %1[%c0][] : memref<?xi8> to memref<16x16xf32>
  %3 = memref.cast %2 : memref<16x16xf32> to memref<?x?xf32>

  // CHECK:  linalg.matmul ins({{.*}}memref<16x16xf32>, memref<16x16xf32>) outs({{.*}}memref<16x16xf32>)
  linalg.matmul ins(%3, %3: memref<?x?xf32>, memref<?x?xf32>)
               outs(%3: memref<?x?xf32>)
  return %3: memref<?x?xf32>
}

// -----

#map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @memref_cast_into_tiled_loop(
func @memref_cast_into_tiled_loop(%arg0: memref<192xf32>)  {
  %0 = memref.cast %arg0
    : memref<192xf32> to memref<192xf32, #map>
  %cst = constant 0.000000e+00 : f32
  %c24 = constant 24 : index
  %c0 = constant 0 : index
  %c192 = constant 192 : index
  // CHECK: linalg.tiled_loop
  // CHECK-SAME: outs (%{{.*}} = %{{.*}}: memref<192xf32>)
  linalg.tiled_loop (%arg3) = (%c0) to (%c192) step (%c24)
    outs (%out = %0: memref<192xf32, #map>) {
    %14 = affine.min affine_map<(d0) -> (-d0 + 192, 24)>(%arg3)
    %16 = memref.subview %out[%arg3] [%14] [1]
      : memref<192xf32, #map> to memref<?xf32, #map>
    linalg.fill(%16, %cst) : memref<?xf32, #map>, f32
    linalg.yield
  }
  return
}

// -----

// CHECK-LABEL: zero_rank_reshape_multi
func @zero_rank_reshape_multi(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: return %arg0
  %0 = linalg.tensor_reshape %arg0 [] : tensor<f32> into tensor<1xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1]] : tensor<1xf32> into tensor<1x1xf32>
  %2 = linalg.tensor_reshape %1 [] : tensor<1x1xf32> into tensor<f32>
  return %2 : tensor<f32>
}

// -----

func @collapsing_tensor_reshapes(%arg0 : tensor<?x?x?x?x?xf32>) -> tensor<?x?xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0, 1], [2], [3, 4]]
      : tensor<?x?x?x?x?xf32> into tensor<?x?x?xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1], [2]]
      : tensor<?x?x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: collapsing_tensor_reshapes
//       CHECK:   linalg.tensor_reshape %{{.*}} {{\[}}[0, 1, 2], [3, 4]]
//   CHECK-NOT:   linalg.tensor_reshape

// -----

func @collapsing_tensor_reshapes_to_zero_dim(%arg0 : tensor<1x1x1xf32>)
                                             -> tensor<f32> {
  %0 = linalg.tensor_reshape %arg0 [[0, 1, 2]]
      : tensor<1x1x1xf32> into tensor<1xf32>
  %1 = linalg.tensor_reshape %0 [] : tensor<1xf32> into tensor<f32>
  return %1 : tensor<f32>
}
// CHECK-LABEL: collapsing_tensor_reshapes_to_zero
//       CHECK:   linalg.tensor_reshape %{{.*}} []
//  CHECK-SAME:     tensor<1x1x1xf32> into tensor<f32>

// -----

func @collapsing_memref_reshapes_to_zero_dim(%arg0 : memref<1x1x1xf32>)
                                             -> memref<f32> {
  %0 = linalg.reshape %arg0 [[0, 1, 2]]
      : memref<1x1x1xf32> into memref<1xf32>
  %1 = linalg.reshape %0 [] : memref<1xf32> into memref<f32>
  return %1 : memref<f32>
}
// CHECK-LABEL: collapsing_memref_reshapes_to_zero
//       CHECK:   linalg.reshape %{{.*}} []
//  CHECK-SAME:     memref<1x1x1xf32> into memref<f32>

// -----

func @expanding_tensor_reshapes(%arg0 : tensor<?x?xf32>) -> tensor<?x6x4x?x5xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0, 1], [2]]
      : tensor<?x?xf32> into tensor<?x4x?xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1], [2], [3, 4]]
      : tensor<?x4x?xf32> into tensor<?x6x4x?x5xf32>
  return %1 : tensor<?x6x4x?x5xf32>
}
// CHECK-LABEL: expanding_tensor_reshapes
//       CHECK:   linalg.tensor_reshape %{{.*}} {{\[}}[0, 1, 2], [3, 4]]
//   CHECK-NOT:   linalg.tensor_reshape

// -----

func @collapsing_memref_reshapes(%arg0 : memref<?x?x?x?x?xf32>) -> memref<?x?xf32>
{
  %0 = linalg.reshape %arg0 [[0, 1], [2], [3, 4]]
      : memref<?x?x?x?x?xf32> into memref<?x?x?xf32>
  %1 = linalg.reshape %0 [[0, 1], [2]]
      : memref<?x?x?xf32> into memref<?x?xf32>
  return %1 : memref<?x?xf32>
}
// CHECK-LABEL: collapsing_memref_reshapes
//       CHECK:   linalg.reshape %{{.*}} {{\[}}[0, 1, 2], [3, 4]]
//   CHECK-NOT:   linalg.reshape

// -----

func @expanding_memref_reshapes(%arg0 : memref<?x?xf32>) -> memref<?x6x4x5x?xf32>
{
  %0 = linalg.reshape %arg0 [[0, 1], [2]]
      : memref<?x?xf32> into memref<?x4x?xf32>
  %1 = linalg.reshape %0 [[0, 1], [2], [3, 4]]
      : memref<?x4x?xf32> into memref<?x6x4x5x?xf32>
  return %1 : memref<?x6x4x5x?xf32>
}
// CHECK-LABEL: expanding_memref_reshapes
//       CHECK:   linalg.reshape %{{.*}} {{\[}}[0, 1, 2], [3, 4]]
//   CHECK-NOT:   linalg.reshape

// -----

func @expanding_tensor_reshapes_to_zero_dim(%arg0 : tensor<f32>)
                                             -> tensor<1x1x1xf32> {
  %0 = linalg.tensor_reshape %arg0 [] : tensor<f32> into tensor<1xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1, 2]] 
      : tensor<1xf32> into tensor<1x1x1xf32>
  return %1 : tensor<1x1x1xf32>
}
// CHECK-LABEL: expanding_tensor_reshapes_to_zero
//       CHECK:   linalg.tensor_reshape %{{.*}} []
//  CHECK-SAME:     tensor<f32> into tensor<1x1x1xf32>

// -----

func @expanding_memref_reshapes_to_zero_dim(%arg0 : memref<f32>)
                                             -> memref<1x1x1xf32> {
  %0 = linalg.reshape %arg0 [] : memref<f32> into memref<1xf32>
  %1 = linalg.reshape %0 [[0, 1, 2]]
      : memref<1xf32> into memref<1x1x1xf32>
  return %1 : memref<1x1x1xf32>
}
// CHECK-LABEL: expanding_memref_reshapes_to_zero
//       CHECK:   linalg.reshape %{{.*}} []
//  CHECK-SAME:     memref<f32> into memref<1x1x1xf32>

// -----

func @fold_tensor_reshape(%arg0 : tensor<12x4xf32>) -> tensor<12x4xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0, 1], [2]]
      : tensor<12x4xf32> into tensor<3x4x4xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1], [2]]
      : tensor<3x4x4xf32> into tensor<12x4xf32>
  return %1 : tensor<12x4xf32>
}
// CHECK-LABEL: @fold_tensor_reshape
//   CHECK-NOT:   linalg.tensor_reshape

// -----

func @fold_tensor_reshape_dynamic(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0, 1], [2]]
      : tensor<?x?xf32> into tensor<?x4x?xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1], [2]]
      : tensor<?x4x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: @fold_tensor_reshape_dynamic
//   CHECK-NOT:   linalg.tensor_reshape

// -----

func @fold_memref_reshape(%arg0 : memref<12x4xf32>) -> memref<12x4xf32>
{
  %0 = linalg.reshape %arg0 [[0, 1], [2]]
      : memref<12x4xf32> into memref<3x4x4xf32>
  %1 = linalg.reshape %0 [[0, 1], [2]]
      : memref<3x4x4xf32> into memref<12x4xf32>
  return %1 : memref<12x4xf32>
}
// CHECK-LABEL: @fold_memref_reshape
//   CHECK-NOT:   linalg.reshape

// -----

func @fold_memref_reshape_dynamic(%arg0 : memref<?x?xf32>) -> memref<?x?xf32>
{
  %0 = linalg.reshape %arg0 [[0, 1], [2]]
      : memref<?x?xf32> into memref<?x4x?xf32>
  %1 = linalg.reshape %0 [[0, 1], [2]]
      : memref<?x4x?xf32> into memref<?x?xf32>
  return %1 : memref<?x?xf32>
}
// CHECK-LABEL: @fold_memref_reshape_dynamic
//   CHECK-NOT:   linalg.reshape

// -----

func @reshape_collapse(%arg0 : tensor<2x3x4x5x6x7x8xf32>) -> tensor<24x5x42x8xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0, 1, 2, 3, 4, 5, 6]]
      : tensor<2x3x4x5x6x7x8xf32> into tensor<40320xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1, 2, 3]]
      : tensor<40320xf32> into tensor<24x5x42x8xf32>
  return %1 : tensor<24x5x42x8xf32>
}
//      CHECK: func @reshape_collapse
// CHECK-SAME:   %[[ARG0:.+]]: tensor<2x3x4x5x6x7x8xf32>
//      CHECK:   %[[RESULT:.+]] = linalg.tensor_reshape %[[ARG0]]
// CHECK-SAME:     [0, 1, 2], [3], [4, 5], [6]
//      CHECK:   return %[[RESULT]]

// -----

func @reshape_expand(%arg0 : tensor<24x5x42x8xf32>) -> tensor<2x3x4x5x6x7x8xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0, 1, 2, 3]]
      : tensor<24x5x42x8xf32> into tensor<40320xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1, 2, 3, 4, 5, 6]]
      : tensor<40320xf32> into tensor<2x3x4x5x6x7x8xf32>
  return %1 : tensor<2x3x4x5x6x7x8xf32>
}
//      CHECK: func @reshape_expand
// CHECK-SAME:   %[[ARG0:.+]]: tensor<24x5x42x8xf32>
//      CHECK:   %[[RESULT:.+]] = linalg.tensor_reshape %[[ARG0]]
// CHECK-SAME:     [0, 1, 2], [3], [4, 5], [6]
//      CHECK:   return %[[RESULT]]

// -----

func @expand_reshape_1D(%arg0 : tensor<2048xf32>) -> tensor<4x512xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0, 1, 2, 3]]
    : tensor<2048xf32> into tensor<1x4x1x512xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1, 2], [3]]
    : tensor<1x4x1x512xf32> into tensor<4x512xf32>
  return %1 : tensor<4x512xf32>
}
//       CHECK: func @expand_reshape_1D
//       CHECK: linalg.tensor_reshape %{{.*}} {{\[}}[0, 1]]
//  CHECK-SAME:   tensor<2048xf32> into tensor<4x512xf32>

// -----

func @fold_reshape_1D(%arg0 : tensor<4x512xf32>) -> tensor<2048xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0, 1, 2], [3]]
    : tensor<4x512xf32> into tensor<1x4x1x512xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1, 2, 3]]
    : tensor<1x4x1x512xf32> into tensor<2048xf32>
  return %1 : tensor<2048xf32>
}
//       CHECK: func @fold_reshape_1D
//       CHECK: linalg.tensor_reshape %{{.*}} {{\[}}[0, 1]]
//  CHECK-SAME:   tensor<4x512xf32> into tensor<2048xf32>

// -----

func @fold_reshape_unit_dims(%arg0 : tensor<2048x1x1xf32>) -> tensor<4x512x1x1xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0, 1, 2, 3], [4], [5]]
    : tensor<2048x1x1xf32> into tensor<1x4x1x512x1x1xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1, 2], [3], [4], [5]]
    : tensor<1x4x1x512x1x1xf32> into tensor<4x512x1x1xf32>
  return %1 : tensor<4x512x1x1xf32>
}
//       CHECK: func @fold_reshape_unit_dims
//       CHECK: linalg.tensor_reshape %{{.*}} {{\[}}[0, 1], [2], [3]]
//  CHECK-SAME:   tensor<2048x1x1xf32> into tensor<4x512x1x1xf32>

// -----

func @expand_reshape_unit_dims(%arg0 : tensor<2048x1x2048xf32>) -> tensor<4x512x1x512x4xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0, 1, 2, 3, 4], [5], [6, 7, 8]]
    : tensor<2048x1x2048xf32> into tensor<1x4x1x512x1x1x512x1x4xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1, 2], [3, 4], [5], [6, 7], [8]]
    : tensor<1x4x1x512x1x1x512x1x4xf32> into tensor<4x512x1x512x4xf32>
  return %1 : tensor<4x512x1x512x4xf32>
}
//       CHECK: func @expand_reshape_unit_dims
//       CHECK: linalg.tensor_reshape %{{.*}} {{\[}}[0, 1], [2], [3, 4]]
//  CHECK-SAME:   tensor<2048x1x2048xf32> into tensor<4x512x1x512x4xf32>

// -----

func @fold_reshape_trailing_unit_dims(%arg0: tensor<2xf32>) -> tensor<2x1xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0, 1, 2]]
      : tensor<2xf32> into tensor<2x1x1xf32>
  %1 = linalg.tensor_reshape %0 [[0], [1, 2]]
      : tensor<2x1x1xf32> into tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}
//       CHECK: func @fold_reshape_trailing_unit_dims
//       CHECK: linalg.tensor_reshape %{{.*}} {{\[}}[0, 1]]
//  CHECK-SAME:   tensor<2xf32> into tensor<2x1xf32>

// -----

func @collapse_reshape_unit_dims_dynamic(%arg0 : tensor<?x1x?x1x1x?x?x1x1xf32>) -> tensor<?x?x?x?xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0], [1, 2], [3], [4], [5], [6, 7, 8]]
    : tensor<?x1x?x1x1x?x?x1x1xf32> into tensor<?x?x1x1x?x?xf32>
  %1 = linalg.tensor_reshape %0 [[0], [1], [2, 3, 4], [5]]
    : tensor<?x?x1x1x?x?xf32> into tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}
//       CHECK: func @collapse_reshape_unit_dims_dynamic
//       CHECK: linalg.tensor_reshape
//  CHECK-SAME:   [0], [1, 2], [3, 4, 5], [6, 7, 8]
//  CHECK-SAME:   tensor<?x1x?x1x1x?x?x1x1xf32> into tensor<?x?x?x?xf32>

// -----

func @fold_reshape_trailing_unit_dims(%arg0: tensor<2xf32>) -> tensor<2x1xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0, 1, 2]]
      : tensor<2xf32> into tensor<2x1x1xf32>
  %1 = linalg.tensor_reshape %0 [[0], [1, 2]]
      : tensor<2x1x1xf32> into tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}
//       CHECK: func @fold_reshape_trailing_unit_dims
//       CHECK: linalg.tensor_reshape %{{.*}} {{\[}}[0, 1]]
//  CHECK-SAME:   tensor<2xf32> into tensor<2x1xf32>

// -----

func @fold_reshape_trailing_unit_dims_dynamic(%arg0: tensor<1x1x?x1x1x1xf32>) -> tensor<?xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0, 1, 2], [3], [4], [5]]
      : tensor<1x1x?x1x1x1xf32> into tensor<?x1x1x1xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1, 2, 3]]
      : tensor<?x1x1x1xf32> into tensor<?xf32>
  return %1 : tensor<?xf32>
}
//       CHECK: func @fold_reshape_trailing_unit_dims_dynamic
//       CHECK: linalg.tensor_reshape %{{.*}} {{\[}}[0, 1, 2, 3, 4, 5]]
//  CHECK-SAME:   tensor<1x1x?x1x1x1xf32> into tensor<?xf32>

// -----

func @no_fold_reshapes(%arg0 : tensor<?x?x?xf32>) -> tensor<?x?xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0], [1], [2, 3]]
      : tensor<?x?x?xf32> into tensor<?x?x1x?xf32>
  %1 = linalg.tensor_reshape %0 [[0], [1, 2, 3]]
      : tensor<?x?x1x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @no_fold_reshapes
//       CHECK:   linalg.tensor_reshape
//       CHECK:   linalg.tensor_reshape

// -----

func @no_fold_reshape_incompatible(%arg0 : tensor<4x6x8xf32>) -> tensor<2x6x16xf32>
{
  %0 = linalg.tensor_reshape %arg0 [[0, 1], [2, 3], [4]]
      : tensor<4x6x8xf32> into tensor<2x2x3x2x8xf32>
  %1 = linalg.tensor_reshape %0 [[0], [1, 2], [3, 4]]
      : tensor<2x2x3x2x8xf32> into tensor<2x6x16xf32>
  return %1 : tensor<2x6x16xf32>
}
// CHECK-LABEL: func @no_fold_reshape_incompatible
//       CHECK:   linalg.tensor_reshape
//       CHECK:   linalg.tensor_reshape

// -----

func @no_fold_reshape_empty_expr(%arg0: tensor<3x2x2xf32>) -> tensor<12x1xf32> {
  %0 = linalg.tensor_reshape %arg0 [[0], [1], [2, 3]]
      : tensor<3x2x2xf32> into tensor<3x2x2x1xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1, 2], [3]]
      : tensor<3x2x2x1xf32> into tensor<12x1xf32>
  return %1 : tensor<12x1xf32>
}
//      CHECK: func @no_fold_reshape_empty_expr
// CHECK-SAME:    %[[ARG0:.+]]: tensor<3x2x2xf32>
//      CHECK:    %[[RARG0:.+]] = linalg.tensor_reshape %[[ARG0]]
// CHECK-SAME:      [0], [1], [2, 3]
//      CHECK:    %[[RES:.+]] = linalg.tensor_reshape %[[RARG0]]
// CHECK-SAME:      [0, 1, 2], [3]
//      CHECK:    return %[[RES:.+]] : tensor<12x1xf32>

// -----

#accesses = [
  affine_map<(i) -> (i)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

func @dce_zero_memref(%arg0 : memref<0xf32>, %arg1: tensor<0xf32>) -> tensor<0xf32> {
  // memref<0x32> is expected to be dce'ed
  linalg.copy(%arg0, %arg0): memref<0xf32>, memref<0xf32>

  // tensor<0xf32> cannot be dce'ed
  %1 = linalg.generic #trait outs(%arg1 : tensor<0xf32>) {
  ^bb(%0: f32) :
    linalg.yield %0 : f32
  } -> tensor<0xf32>

  return %1: tensor<0xf32>
}
// CHECK-LABEL: @dce_zero_memref
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<0xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<0xf32>
//   CHECK-NOT:   linalg.copy
//  CHECK-NEXT:   return %[[ARG1]]

// -----

func @reshape_splat_constant_int32() -> tensor<2x4x2xi32>
{
  %c0 = constant dense<42> : tensor<2x8xi32>
  %0 = linalg.tensor_reshape %c0 [[0], [1, 2]]
      : tensor<2x8xi32> into tensor<2x4x2xi32>
  return %0 : tensor<2x4x2xi32>
}
// CHECK-LABEL: @reshape_splat_constant_int32
//       CHECK:   %[[CST:.*]] = constant dense<{{.*}}> : tensor<2x4x2xi32>
//   CHECK-NOT:   linalg.tensor_reshape
//       CHECK:   return %[[CST]]

func @reshape_splat_constant_int16() -> tensor<2x4x2xi16>
{
  %c0 = constant dense<42> : tensor<2x8xi16>
  %0 = linalg.tensor_reshape %c0 [[0], [1, 2]]
      : tensor<2x8xi16> into tensor<2x4x2xi16>
  return %0 : tensor<2x4x2xi16>
}
// CHECK-LABEL: @reshape_splat_constant_int16
//       CHECK:   %[[CST:.*]] = constant dense<{{.*}}> : tensor<2x4x2xi16>
//   CHECK-NOT:   linalg.tensor_reshape
//       CHECK:   return %[[CST]]

func @reshape_splat_constant_float32() -> tensor<2x4x2xf32>
{
  %c0 = constant dense<42.0> : tensor<2x8xf32>
  %0 = linalg.tensor_reshape %c0 [[0], [1, 2]]
      : tensor<2x8xf32> into tensor<2x4x2xf32>
  return %0 : tensor<2x4x2xf32>
}
// CHECK-LABEL: @reshape_splat_constant_float32
//       CHECK:   %[[CST:.*]] = constant dense<{{.*}}> : tensor<2x4x2xf32>
//   CHECK-NOT:   linalg.tensor_reshape
//       CHECK:   return %[[CST]]

func @reshape_splat_constant_float64() -> tensor<2x4x2xf64>
{
  %c0 = constant dense<42.0> : tensor<2x8xf64>
  %0 = linalg.tensor_reshape %c0 [[0], [1, 2]]
      : tensor<2x8xf64> into tensor<2x4x2xf64>
  return %0 : tensor<2x4x2xf64>
}
// CHECK-LABEL: @reshape_splat_constant_float64
//       CHECK:   %[[CST:.*]] = constant dense<{{.*}}> : tensor<2x4x2xf64>
//   CHECK-NOT:   linalg.tensor_reshape
//       CHECK:   return %[[CST]]

// -----

// CHECK-LABEL: func @tensor.cast(
func @tensor.cast(%a : tensor<3x4xf32>, %b : tensor<4x?xf32>, %c : tensor<3x?xf32>)
  -> tensor<3x?xf32>
{
  %ta = tensor.cast %a : tensor<3x4xf32> to tensor<?x?xf32>
  %tb = tensor.cast %b : tensor<4x?xf32> to tensor<?x?xf32>
  %tc = tensor.cast %c : tensor<3x?xf32> to tensor<?x?xf32>

  //      CHECK:  linalg.matmul ins({{.*}}tensor<3x4xf32>, tensor<4x?xf32>)
  // CHECK-SAME:    outs({{.*}}tensor<3x?xf32>) -> tensor<3x?xf32>
  %0 = linalg.matmul ins(%ta, %tb: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%tc: tensor<?x?xf32>) -> tensor<?x?xf32>

  %1 = tensor.cast %0 : tensor<?x?xf32> to tensor<3x?xf32>

  return %1: tensor<3x?xf32>
}

// -----

// CHECK-LABEL: func @linalg_effects(
//  CHECK-SAME:     %[[A:[a-z0-9]*]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[B:[a-z0-9]*]]: memref<?x?xf32>
//  CHECK-SAME:     %[[C:[a-z0-9]*]]: tensor<?x?xf32>
func @linalg_effects(%a : tensor<?x?xf32>, %b : memref<?x?xf32>, %c : tensor<?x?xf32>) {
  // CHECK-NOT:   %{{.*}} = linalg.matmul
  %t = linalg.matmul ins(%a, %b : tensor<?x?xf32>, memref<?x?xf32>)
                    outs(%c : tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK:   linalg.matmul
  linalg.matmul ins(%a, %c : tensor<?x?xf32>, tensor<?x?xf32>)
               outs(%b : memref<?x?xf32>)
  return
}

// -----

func @init_tensor_canonicalize() -> (tensor<4x5x?xf32>) {
  %c6 = constant 6 : index
  %0 = linalg.init_tensor [4, 5, %c6] : tensor<4x5x?xf32>
  return %0 : tensor<4x5x?xf32>
}
// CHECK: func @init_tensor_canonicalize
// CHECK:   %[[T0:.+]] = linalg.init_tensor [4, 5, 6] : tensor<4x5x6xf32>
// CHECK:   %[[T1:.+]] = tensor.cast %[[T0]] : tensor<4x5x6xf32> to tensor<4x5x?xf32>
// CHECK:   return %[[T1]]

// -----

func @init_tensor_static_dim() -> (index, index) {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c6 = constant 6 : index
  %0 = linalg.init_tensor [4, 5, %c6] : tensor<4x5x?xf32>
  %1 = memref.dim %0, %c2 : tensor<4x5x?xf32>
  %2 = memref.dim %0, %c0 : tensor<4x5x?xf32>
  return %1, %2 : index, index
}
//      CHECK: func @init_tensor_static_dim
//  CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//  CHECK-DAG:   %[[C6:.+]] = constant 6 : index
//      CHECK:   return %[[C6]], %[[C4]]

// -----

func @init_tensor_dynamic_dim(%arg0 : index) -> (index) {
  %c2 = constant 2 : index
  %0 = linalg.init_tensor [4, 5, %arg0] : tensor<4x5x?xf32>
  %1 = memref.dim %0, %c2 : tensor<4x5x?xf32>
  return %1 : index
}
//      CHECK: func @init_tensor_dynamic_dim
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
//      CHECK:   return %[[ARG0]]

// -----

func @init_tensor_dynamic_dim2(%arg0 : index, %arg1 : index) -> (index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = linalg.init_tensor [%arg0, %arg1] : tensor<?x?xf32>
  %1 = memref.dim %0, %c0 : tensor<?x?xf32>
  %2 = memref.dim %0, %c1 : tensor<?x?xf32>
  return %1, %2 : index, index
}
//      CHECK: func @init_tensor_dynamic_dim2
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//      CHECK:   return %[[ARG0]], %[[ARG1]]

// -----

func @remove_dim_result_uses
  (%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
   %arg2 : tensor<?x?xf32>) -> (index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d2, d1)>,
                      affine_map<(d0, d1, d2) -> (d0 + d1, d1 - d0)>],
     iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) {
    ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
      %1 = mulf %arg3, %arg4 : f32
      %2 = addf %1, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<?x?xf32>
  %3 = memref.dim %0, %c0 : tensor<?x?xf32>
  %4 = memref.dim %0, %c1 : tensor<?x?xf32>
  return %3, %4 : index, index
}
//       CHECK: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (-s0 + s1)>
//       CHECK: func @remove_dim_result_uses
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[T0:.+]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[T1:.+]] = memref.dim %[[ARG1]], %[[C1]]
//       CHECK:   %[[T2:.+]] = affine.apply #[[MAP0]]()[%[[T0]], %[[T1]]]
//   CHECK-DAG:   %[[T3:.+]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[T4:.+]] = memref.dim %[[ARG1]], %[[C1]]
//       CHECK:   %[[T5:.+]] = affine.apply #[[MAP1]]()[%[[T3]], %[[T4]]]
//       CHECK:   return %[[T2]], %[[T5]]

// -----

func @remove_dim_result_uses_outs
  (%arg0 : tensor<?xf32>, %arg1 : index) -> (index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %d0 = memref.dim %arg0, %c0 : tensor<?xf32>
  %0 = linalg.init_tensor [%d0, %arg1] : tensor<?x?xf32>
  %1 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?xf32>) outs(%0 : tensor<?x?xf32>) {
    ^bb0(%arg2: f32, %arg3: f32) :
      linalg.yield %arg2 : f32
    } -> tensor<?x?xf32>
  %2 = memref.dim %1, %c1 : tensor<?x?xf32>
  return %2 : index
}
//      CHECK: func @remove_dim_result_uses_outs
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//      CHECK:   return %[[ARG1]]

// -----

func @remove_dim_result_uses_sequence
  (%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
   %arg2 : tensor<?x?xf32>) -> (index, index, index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = memref.dim %0, %c0 : tensor<?x?xf32>
  %2 = memref.dim %0, %c1 : tensor<?x?xf32>
  %3 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0)>,
                      affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d2)>],
     iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%0 : tensor<?x?xf32>) {
    ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
      %4 = mulf %arg3, %arg4 : f32
      %5 = addf %4, %arg5 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  %6 = memref.dim %3, %c0 : tensor<?x?xf32>
  %7 = memref.dim %3, %c1 : tensor<?x?xf32>
  return %1, %2, %6, %7 : index, index, index, index
}
// CHECK-LABEL: func @remove_dim_result_uses_sequence
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[T0:.+]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[T1:.+]] = memref.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[T2:.+]] = memref.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[T3:.+]] = memref.dim %[[ARG1]], %[[C1]]
//       CHECK:   return %[[T0]], %[[T1]], %[[T2]], %[[T3]]

// -----

func @keep_result_dim_uses_sequence2
  (%arg0 : tensor<?xf32>, %arg1 : index) -> (index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %d0 = memref.dim %arg0, %c0 : tensor<?xf32>
  %0 = linalg.init_tensor [%d0, %arg1] : tensor<?x?xf32>
  %1 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?xf32>) outs(%0 : tensor<?x?xf32>) {
    ^bb0(%arg2: f32, %arg3 : f32):
      linalg.yield %arg2 : f32
    } -> tensor<?x?xf32>
  %2 = memref.dim %1, %c0 : tensor<?x?xf32>
  %3 = memref.dim %1, %c1 : tensor<?x?xf32>
  return %2, %3 : index, index
}
//       CHECK: func @keep_result_dim_uses_sequence2
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[T0:.+]] = memref.dim %[[ARG0]], %[[C0]]
//       CHECK:   return %[[T0]], %[[ARG1]]

// -----

#map = affine_map<(d0) -> (d0)>

func @init_tensor_dim_of_linalg_result(%arg_0 : tensor<?xf32>,
    %arg_1: tensor<?xf32>) -> (index, index) {
  %0, %1 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%arg_0 : tensor<?xf32>)
    outs(%arg_0, %arg_1 : tensor<?xf32>, tensor<?xf32>) {
  ^bb0(%in: f32, %out_0: f32, %out_1: f32):
    linalg.yield %in, %in : f32, f32
  } -> tensor<?xf32>, tensor<?xf32>

  %c0 = constant 0 : index
  %num_elem_0 = memref.dim %0, %c0 : tensor<?xf32>

  %num_elem_1 = memref.dim %1, %c0 : tensor<?xf32>
  return %num_elem_0, %num_elem_1 : index, index
}
//      CHECK: func @init_tensor_dim_of_linalg_result(
// CHECK-SAME:   %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<?xf32>
// CHECK-SAME:   %[[ARG_1:[a-zA-Z0-9_]+]]: tensor<?xf32>)
//      CHECK:   %[[R0:.+]] = memref.dim %[[ARG_0]]
//      CHECK:   %[[R1:.+]] = memref.dim %[[ARG_0]]
//      CHECK:   return %[[R0]], %[[R1]]

// -----

func @init_tensor_reshape_expansion(%arg0 : index) -> tensor<2x3x5x4x?x7xf32> {
  %0 = linalg.init_tensor [6, 5, %arg0] : tensor<6x5x?xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1], [2], [3, 4, 5]]
      : tensor<6x5x?xf32> into tensor<2x3x5x4x?x7xf32>
  return %1 : tensor<2x3x5x4x?x7xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 28)>
//      CHECK: func @init_tensor_reshape_expansion
// CHECK-SAME:   %[[ARG0:.+]]: index
//      CHECK:   %[[T0:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//      CHECK:   %[[T1:.+]] = linalg.init_tensor [2, 3, 5, 4, %[[T0]], 7]
//      CHECK:   return %[[T1]]

// -----

func @init_tensor_reshape_collapse(%arg0 : index) -> tensor<6x5x?xf32> {
  %0 = linalg.init_tensor [2, 3, 5, 4, %arg0, 7] : tensor<2x3x5x4x?x7xf32>
  %1 = linalg.tensor_reshape %0 [[0, 1], [2], [3, 4, 5]]
      : tensor<2x3x5x4x?x7xf32> into tensor<6x5x?xf32>
  return %1 : tensor<6x5x?xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 28)>
//      CHECK: func @init_tensor_reshape_collapse
// CHECK-SAME:   %[[ARG0:.+]]: index
//      CHECK:   %[[T0:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//      CHECK:   %[[T1:.+]] = linalg.init_tensor [6, 5, %[[T0]]]
//      CHECK:   return %[[T1]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @remove_no_op(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>)
  -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %0 = memref.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = memref.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = memref.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = linalg.init_tensor [%0, %1, %2] : tensor<?x?x?xf32>
  %4, %5 = linalg.generic {
    indexing_maps = [#map, #map, #map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs(%3, %3 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  ^bb0(%arg2 : f32, %arg3 : f32, %arg4 : f32, %arg5 : f32):
    linalg.yield %arg3, %arg2 : f32, f32
  } -> tensor<?x?x?xf32>, tensor<?x?x?xf32>
  return %4, %5 : tensor<?x?x?xf32>, tensor<?x?x?xf32>
}
// CHECK-LABEL: func @remove_no_op
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//       CHECK:     return %[[ARG1]], %[[ARG0]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func @keep_not_noop(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %cst = constant 1.000000e+00 : f32
  %0 = memref.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = memref.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  br ^bb1(%cst : f32)

^bb1(%arg1 : f32):
  %3 = linalg.generic
    {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg2: f32, %arg3 : f32):
      linalg.yield %arg1 : f32
    } -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}
// CHECK-LABEL: func @keep_not_noop
//       CHECK:   %[[RESULT:.+]] = linalg.generic
//       CHECK:   return %[[RESULT]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func @keep_not_noop(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %cst = constant 1.000000e+00 : f32
  %0 = memref.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = memref.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  br ^bb1(%cst : f32)

^bb1(%arg2 : f32):
  %3:2 = linalg.generic
    {indexing_maps = [#map, #map, #map, #map],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%2, %2 : tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4 : f32, %arg5 : f32, %arg6 : f32):
      linalg.yield %arg2, %arg4 : f32, f32
    } -> tensor<?x?xf32>, tensor<?x?xf32>
  return %3#0, %3#1 : tensor<?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func @keep_not_noop
//       CHECK:   %[[RESULT:.+]]:2 = linalg.generic
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func @fold_init_tensor_with_subtensor
  (%arg0 : index, %arg1 : index) -> tensor<5x?x20xf32>
{
  %0 = linalg.init_tensor[%arg0, 10, 40] : tensor<?x10x40xf32>
  %1 = subtensor %0[0, 0, 0] [5, %arg1, 20] [1, 1, 1]
    : tensor<?x10x40xf32> to tensor<5x?x20xf32>
  return %1 : tensor<5x?x20xf32>
}
//      CHECK: func @fold_init_tensor_with_subtensor
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//      CHECK:   %[[T0:.+]] = linalg.init_tensor [5, %[[ARG1]], 20]
//      CHECK:   return %[[T0]]

// -----

#accesses = [
  affine_map<(i, j) -> (i, j)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel"]
}

// CHECK-LABEL: func @dead_linalg_tensor
//   CHECK-NOT:   linalg.fill
//   CHECK-NOT:   linalg.matmul
//   CHECK-NOT:   linalg.generic
//   CHECK-NOT:   linalg.pad_tensor
//       CHECK:   return
func @dead_linalg_tensor(%arg0 : tensor<7x7xi32>, %arg1 : tensor<7x7xf32>,
                         %arg2: tensor<?x?xf32>, %high : index) {
  %c0_i32 = constant 0 : i32
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.fill(%arg0, %c0_i32) : tensor<7x7xi32>, i32 -> tensor<7x7xi32>
  %1 = linalg.matmul ins(%arg1, %arg1: tensor<7x7xf32>, tensor<7x7xf32>)
                     outs(%arg1: tensor<7x7xf32>) -> tensor<7x7xf32>
  %2 = linalg.generic #trait outs(%arg0 : tensor<7x7xi32>) {
  ^bb(%3: i32) :
    linalg.yield %3 : i32
  } -> tensor<7x7xi32>
  %3 = linalg.pad_tensor %arg2 low[%c0, %c0] high[%high, %high] {
        ^bb0(%arg9: index, %arg10: index):  // no predecessors
          linalg.yield %cst : f32
  } : tensor<?x?xf32> to tensor<2x4xf32>
  return
}

// -----

func @dim_reshape_expansion(%arg0 : tensor<6x5x?xf32>) -> (index, index, index)
{
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %0 = linalg.tensor_reshape %arg0 [[0, 1], [2], [3, 4, 5]]
      : tensor<6x5x?xf32> into tensor<2x3x5x4x?x7xf32>
  %1 = memref.dim %0, %c1 : tensor<2x3x5x4x?x7xf32>
  %2 = memref.dim %0, %c3 : tensor<2x3x5x4x?x7xf32>
  %3 = memref.dim %0, %c4 : tensor<2x3x5x4x?x7xf32>
  return %1, %2, %3 : index, index, index
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 28)>
//      CHECK: func @dim_reshape_expansion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<6x5x?xf32>
//  CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//  CHECK-DAG:   %[[C3:.+]] = constant 3 : index
//  CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//      CHECK:   %[[D0:.+]] = memref.dim %[[ARG0]], %[[C2]]
//      CHECK:   %[[D1:.+]] = affine.apply #[[MAP]]()[%[[D0]]]
//      CHECK:   return %[[C3]], %[[C4]], %[[D1]]

// -----

func @dim_reshape_collapse(%arg0 : tensor<2x3x5x4x?x7xf32>) -> (index, index)
{
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %0 = linalg.tensor_reshape %arg0 [[0, 1], [2], [3, 4, 5]]
      : tensor<2x3x5x4x?x7xf32> into tensor<6x5x?xf32>
  %1 = memref.dim %0, %c1 : tensor<6x5x?xf32>
  %2 = memref.dim %0, %c2 : tensor<6x5x?xf32>
  return %1, %2 : index, index
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 28)>
//      CHECK: func @dim_reshape_collapse
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x3x5x4x?x7xf32>
//  CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//  CHECK-DAG:   %[[C5:.+]] = constant 5 : index
//      CHECK:   %[[D0:.+]] = memref.dim %[[ARG0]], %[[C4]]
//      CHECK:   %[[D1:.+]] = affine.apply #[[MAP]]()[%[[D0]]]
//      CHECK:   return %[[C5]], %[[D1]]

// -----

func @propogate_casts(%arg0 : tensor<?x?xf32>, %arg1 : f32, %arg2 : index,
    %arg3 : index) -> tensor<?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c21 = constant 21 : index
  %c42 = constant 42 : index
  %0 = linalg.init_tensor [%c21, %c42] : tensor<?x?xf32>
  %1 = linalg.fill(%0, %arg1) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
  %2 = memref.dim %arg0, %c0 : tensor<?x?xf32>
  %3 = memref.dim %arg0, %c1 : tensor<?x?xf32>
  %4 = subtensor_insert %arg0 into %1[%arg2, %arg3] [%2, %3] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
// CHECK-LABEL: func @propogate_casts
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor [21, 42]
//       CHECK:   %[[FILL:.+]] = linalg.fill(%[[INIT]], %{{.+}})
//       CHECK:   %[[INSERTED:.+]] = subtensor_insert %{{.+}} into %[[FILL]]
//       CHECK:   %[[RESULT:.+]] = tensor.cast %[[INSERTED]]
//       CHECK:   return %[[RESULT]]

// -----

// CHECK-LABEL: @self_copy
func @self_copy(%arg0 : memref<2x3x?x4xf32>) {

//   CHECK-NOT: linalg.copy
  linalg.copy(%arg0, %arg0): memref<2x3x?x4xf32>, memref<2x3x?x4xf32>

//   CHECK: return
  return
}

// -----

// CHECK-LABEL: func @fold_fill_reshape()
func @fold_fill_reshape() -> tensor<6x4xf32> {
  %zero = constant 0.0 : f32
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [6, 4] : tensor<6x4xf32>
  %init = linalg.init_tensor [1, 2, 3, 4] : tensor<1x2x3x4xf32>
  // CHECK: %[[FILL:.+]] = linalg.fill(%[[INIT]], %cst) : tensor<6x4xf32>, f32 -> tensor<6x4xf32>
  %fill = linalg.fill(%init, %zero) : tensor<1x2x3x4xf32>, f32 -> tensor<1x2x3x4xf32>
  %reshape = linalg.tensor_reshape %fill [[0, 1, 2], [3]]
      : tensor<1x2x3x4xf32> into tensor<6x4xf32>
  // CHECK: return %[[FILL]] : tensor<6x4xf32>
  return %reshape : tensor<6x4xf32>
}

// -----

//       CHECK: func @fold_fill_reshape_dynamic
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?x?x?xf32>
func @fold_fill_reshape_dynamic(%arg0 : tensor<?x?x?x?x?xf32>) -> tensor<?x?xf32> {
  %zero = constant 0.0 : f32
  // CHECK: %[[RESHAPE:.+]] = linalg.tensor_reshape %[[ARG0]]
  %0 = linalg.fill(%arg0, %zero) : tensor<?x?x?x?x?xf32>, f32 -> tensor<?x?x?x?x?xf32>
  // CHECK: %[[RESULT:.+]] = linalg.fill(%[[RESHAPE]], %{{.+}})
  %1 = linalg.tensor_reshape %0 [[0, 1, 2], [3, 4]]
      : tensor<?x?x?x?x?xf32> into tensor<?x?xf32>
  // CHECK: return %[[RESULT]]
  return %1 : tensor<?x?xf32>
}


// -----

func private @foo(%A: memref<48xf32>, %B: tensor<48xf32>,
                  %C: memref<48xf32>) -> (tensor<48xf32>)

func @fold_tiled_loop_results(%A: memref<48xf32>, %B: tensor<48xf32>,
    %C: memref<48xf32>, %C_tensor: tensor<48xf32>) -> tensor<48xf32> {
  %c0 = constant 0 : index
  %c24 = constant 24 : index
  %c48 = constant 48 : index
  %useful, %useless = linalg.tiled_loop (%i) = (%c0) to (%c48) step (%c24)
      ins (%A_ = %A: memref<48xf32>)
      outs (%B_ = %B: tensor<48xf32>,
            %CT_ = %C_tensor: tensor<48xf32>,
            %C_ = %C: memref<48xf32>) {
        %result = call @foo(%A_, %B_, %C_)
          : (memref<48xf32>, tensor<48xf32>, memref<48xf32>)-> (tensor<48xf32>)
    linalg.yield %result, %CT_ : tensor<48xf32>, tensor<48xf32>
  }
  return %useful : tensor<48xf32>
}

// CHECK-LABEL: func @fold_tiled_loop_results(
// CHECK-SAME:   %[[A:.*]]: [[BUF_TY:memref<48xf32>]], %[[B:.*]]: [[TY:tensor<48xf32>]],
// CHECK-SAME:   %[[C:.*]]: [[BUF_TY]],  %[[C_TENSOR:.*]]: [[TY]]) -> [[TY]] {

// CHECK-DAG:  %[[C0:.*]] = constant 0 : index
// CHECK-DAG:  %[[C24:.*]] = constant 24 : index
// CHECK-DAG:  %[[C48:.*]] = constant 48 : index

// CHECK-NOT: %{{.*}} = linalg.tiled_loop
// CHECK:  %[[RESULT:.*]] = linalg.tiled_loop (%{{.*}}) = (%[[C0]])
// CHECK-SAME: to (%[[C48]]) step (%[[C24]])
// CHECK-SAME: ins (%[[A_:.*]] = %[[A]]: [[BUF_TY]])
// CHECK-SAME: outs (%[[B_:.*]] = %[[B]]: [[TY]], %[[C_:.*]] = %[[C]]: [[BUF_TY]]) {
// CHECK-NEXT:   %[[RES:.*]] = call @foo(%[[A_]], %[[B_]], %[[C_]])
// CHECK-NEXT:   linalg.yield %[[RES]] :

// CHECK: return %[[RESULT]]

// -----

func private @foo(%A: memref<192xf32>, %B: tensor<192xf32>) -> tensor<192xf32>

func @fold_tiled_loop_inputs(%A: memref<192xf32>, %A_tensor: tensor<192xf32>,
                             %B_tensor: tensor<192xf32>) -> tensor<192xf32> {
  %c0 = constant 0 : index
  %c24 = constant 24 : index
  %c192 = constant 192 : index
  %result = linalg.tiled_loop (%i) = (%c0) to (%c192) step (%c24)
      ins (%A_ = %A: memref<192xf32>, %AT_ = %A_tensor: tensor<192xf32>)
      outs (%BT_ = %B_tensor: tensor<192xf32>) {
    %0 = call @foo(%A_, %BT_) : (memref<192xf32>, tensor<192xf32>) -> tensor<192xf32>
    linalg.yield %0 : tensor<192xf32>
  }
  return %result : tensor<192xf32>
}

// CHECK-LABEL: func @fold_tiled_loop_inputs
// CHECK: %[[RESULT:.*]] = linalg.tiled_loop
// CHECK-SAME: ins (%{{.*}} = %{{.*}}: memref<192xf32>)

// CHECK: return %[[RESULT]]

// -----

func @dim_of_pad_op(%arg0 : tensor<2x?x?xf32>, %arg1 : index, %arg2 : index,
    %arg3: f32) -> (index, index, index)
{
   %c0 = constant 0 : index
   %c1 = constant 1 : index
   %c2 = constant 2 : index
   %c3 = constant 3 : index
   %c4 = constant 4 : index
   %c5 = constant 5 : index
   %0 = linalg.pad_tensor %arg0 low[%c3, %arg1, %c4] high[7, %c5, %arg2] {
     ^bb0(%arg4: index, %arg5: index, %arg6: index):
       linalg.yield %arg3 : f32
   } : tensor<2x?x?xf32> to tensor<?x?x?xf32>
   %1 = memref.dim %0, %c0 : tensor<?x?x?xf32>
   %2 = memref.dim %0, %c1 : tensor<?x?x?xf32>
   %3 = memref.dim %0, %c2 : tensor<?x?x?xf32>
   return %1, %2, %3 : index, index, index
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 4)>
//      CHECK: func @dim_of_pad_op
// CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<2x?x?xf32>
// CHECK-SAME:   %[[ARG1:[A-Za-z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[A-Za-z0-9_]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//  CHECK-DAG:   %[[C12:.+]] = constant 12 : index
//      CHECK:   %[[IN_DIM1:.+]] = memref.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[OUT_DIM1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]], %[[IN_DIM1]]]
//      CHECK:   %[[IN_DIM2:.+]] = memref.dim %[[ARG0]], %[[C2]]
//      CHECK:   %[[OUT_DIM2:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[IN_DIM2]]]
//      CHECK:   return %[[C12]], %[[OUT_DIM1]], %[[OUT_DIM2]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func @indexed_generic(%arg0: memref<?x?xindex>, %arg1: memref<?x?xindex>) {
  linalg.indexed_generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<?x?xindex>)
   outs(%arg1 : memref<?x?xindex>) {
  ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
    %0 = addi %arg4, %arg5 : index
    %1 = addi %0, %arg6 : index
    %2 = addi %1, %arg7 : index
    linalg.yield %2 : index
  }
  return
}

//      CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//      CHECK: func @indexed_generic
// CHECK-NEXT:   linalg.generic {
// CHECK-SAME:     indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:      ins(%[[ARG0:[A-Za-z0-9_]+]] : memref<?x?xindex>)
// CHECK-SAME:     outs(%[[ARG1:[A-Za-z0-9_]+]] : memref<?x?xindex>)
//      CHECK:   ^bb0(%[[ARG2:[A-Za-z0-9_]+]]: index, %[[ARG3:[A-Za-z0-9_]+]]: index):
// CHECK-NEXT:     %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-NEXT:     %[[IDX1:.+]] = linalg.index 1 : index
// CHECK-NEXT:     %[[SUM0:.+]] = addi %[[IDX0]], %[[IDX1]] : index
// CHECK-NEXT:     %[[SUM1:.+]] = addi %[[SUM0]], %[[ARG2]] : index
// CHECK-NEXT:     %[[SUM2:.+]] = addi %[[SUM1]], %[[ARG3]] : index
// CHECK-NEXT:     linalg.yield %[[SUM2]] : index
