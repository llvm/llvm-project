// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file | FileCheck %s

// CHECK-LABEL: func @memref_cast(
func.func @memref_cast(%a: index, %b: index) -> memref<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %1 = memref.alloc (%b) : memref<?xi8>
  %2 = memref.view %1[%c0][] : memref<?xi8> to memref<16x16xf32>
  %3 = memref.cast %2 : memref<16x16xf32> to memref<?x?xf32>

  // CHECK:  linalg.matmul ins({{.*}}memref<16x16xf32>, memref<16x16xf32>) outs({{.*}}memref<16x16xf32>)
  linalg.matmul ins(%3, %3: memref<?x?xf32>, memref<?x?xf32>)
               outs(%3: memref<?x?xf32>)
  return %3: memref<?x?xf32>
}

// -----

#accesses = [
  affine_map<(i) -> (i)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

func.func @dce_zero_memref(%arg0 : memref<0xf32>, %arg1: tensor<0xf32>) -> tensor<0xf32> {
  // memref<0x32> is expected to be dce'ed
  memref.copy %arg0, %arg0 : memref<0xf32> to memref<0xf32>

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
//   CHECK-NOT:   memref.copy
//  CHECK-NEXT:   return %[[ARG1]]

// -----

func.func @dce_self_linalg_copy(%arg0 : memref<?xf32>) {
  linalg.copy ins(%arg0: memref<?xf32>) outs(%arg0: memref<?xf32>)
  return
}

// CHECK-LABEL: @dce_self_linalg_copy
//   CHECK-NOT:   copy

// -----

// CHECK-LABEL: func @tensor.cast(
func.func @tensor.cast(%a : tensor<3x4xf32>, %b : tensor<4x?xf32>, %c : tensor<3x?xf32>)
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

// CHECK-LABEL: func @tensor.cast.unranked(
func.func @tensor.cast.unranked(%a : tensor<*xf32>, %b : tensor<*xf32>, %c : tensor<*xf32>)
  -> tensor<*xf32>
{
  //      CHECK:  tensor.cast
  //      CHECK:  tensor.cast
  //      CHECK:  tensor.cast
  %ta = tensor.cast %a : tensor<*xf32> to tensor<?x?xf32>
  %tb = tensor.cast %b : tensor<*xf32> to tensor<?x?xf32>
  %tc = tensor.cast %c : tensor<*xf32> to tensor<?x?xf32>

  //      CHECK:  linalg.matmul ins({{.*}}tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:    outs({{.*}}tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = linalg.matmul ins(%ta, %tb: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%tc: tensor<?x?xf32>) -> tensor<?x?xf32>

  //      CHECK:  tensor.cast
  %1 = tensor.cast %0 : tensor<?x?xf32> to tensor<*xf32>

  return %1: tensor<*xf32>
}

// -----

// CHECK-LABEL: func @linalg_effects(
func.func @linalg_effects(
    %a : tensor<?x?xf32>, %b : tensor<?x?xf32>, %c : tensor<?x?xf32>,
    %d : memref<?x?xf32>, %e : memref<?x?xf32>, %f : memref<?x?xf32>) {
  // CHECK-NOT:   %{{.*}} = linalg.matmul
  %t = linalg.matmul ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%c : tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK:   linalg.matmul
  linalg.matmul ins(%d, %e : memref<?x?xf32>, memref<?x?xf32>)
               outs(%f : memref<?x?xf32>)
  return
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @remove_no_op(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>)
  -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %4, %5 = linalg.generic {
    indexing_maps = [#map, #map, #map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs(%3, %3 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  ^bb0(%arg2 : f32, %arg3 : f32, %arg4 : f32, %arg5 : f32):
    linalg.yield %arg3, %arg2 : f32, f32
  } -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  return %4, %5 : tensor<?x?x?xf32>, tensor<?x?x?xf32>
}
// CHECK-LABEL: func @remove_no_op
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//       CHECK:     return %[[ARG1]], %[[ARG0]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @remove_no_op_mismatched_types(%arg0 : tensor<?x?x?xf32>)
  -> tensor<1x2x3xf32> {
  %out = tensor.empty() : tensor<1x2x3xf32>
  %g = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<?x?x?xf32>)
    outs(%out : tensor<1x2x3xf32>) {
  ^bb0(%arg2 : f32, %arg3 : f32):
    linalg.yield %arg2 : f32
  } -> (tensor<1x2x3xf32>)
  return %g : tensor<1x2x3xf32>
}
// CHECK-LABEL: func @remove_no_op_mismatched_types
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//       CHECK:     %[[CAST:.*]] = tensor.cast %[[ARG0]] : tensor<?x?x?xf32> to tensor<1x2x3xf32>
//       CHECK:     return %[[CAST]]

// -----

#map = affine_map<() -> ()>
func.func @cant_fold_to_tensor_cast(%arg0 : f32) -> tensor<f32> {
  %out = tensor.empty() : tensor<f32>
  %g = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = []
  } ins(%arg0 : f32)
    outs(%out : tensor<f32>) {
  ^bb0(%arg2 : f32, %arg3 : f32):
    linalg.yield %arg2 : f32
  } -> (tensor<f32>)
  return %g : tensor<f32>
}
// CHECK-LABEL: func @cant_fold_to_tensor_cast
//       CHECK:     linalg.generic

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @keep_not_noop(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  cf.br ^bb1(%cst : f32)

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
func.func @keep_not_noop(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  cf.br ^bb1(%cst : f32)

^bb1(%arg2 : f32):
  %3:2 = linalg.generic
    {indexing_maps = [#map, #map, #map, #map],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%2, %2 : tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4 : f32, %arg5 : f32, %arg6 : f32):
      linalg.yield %arg2, %arg4 : f32, f32
    } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  return %3#0, %3#1 : tensor<?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func @keep_not_noop
//       CHECK:   %[[RESULT:.+]]:2 = linalg.generic
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

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
//   CHECK-NOT:   tensor.pad
//       CHECK:   return
func.func @dead_linalg_tensor(%arg0 : tensor<7x7xi32>, %arg1 : tensor<7x7xf32>,
                         %arg2: tensor<?x?xf32>, %high : index) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%c0_i32 : i32) outs(%arg0 : tensor<7x7xi32>) -> tensor<7x7xi32>
  %1 = linalg.matmul ins(%arg1, %arg1: tensor<7x7xf32>, tensor<7x7xf32>)
                     outs(%arg1: tensor<7x7xf32>) -> tensor<7x7xf32>
  %2 = linalg.generic #trait outs(%arg0 : tensor<7x7xi32>) {
  ^bb(%3: i32) :
    linalg.yield %3 : i32
  } -> tensor<7x7xi32>
  %3 = tensor.pad %arg2 low[%c0, %c0] high[%high, %high] {
        ^bb0(%arg9: index, %arg10: index):
          tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<2x4xf32>
  return
}

// -----

func.func @propagate_casts(%arg0 : tensor<?x?xf32>, %arg1 : f32, %arg2 : index,
    %arg3 : index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c21 = arith.constant 21 : index
  %c42 = arith.constant 42 : index
  %0 = tensor.empty(%c21, %c42) : tensor<?x?xf32>
  %1 = linalg.fill ins(%arg1 : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %3 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %4 = tensor.insert_slice %arg0 into %1[%arg2, %arg3] [%2, %3] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
// CHECK-LABEL: func @propagate_casts
//       CHECK:   %[[INIT:.+]] = tensor.empty
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%{{.+}}{{.*}}outs(%[[INIT]]
//       CHECK:   %[[INSERTED:.+]] = tensor.insert_slice %{{.+}} into %[[FILL]]
//       CHECK:   %[[RESULT:.+]] = tensor.cast %[[INSERTED]]
//       CHECK:   return %[[RESULT]]

// -----

// CHECK-LABEL: @self_copy
func.func @self_copy(%arg0 : memref<2x3x?x4xf32>) {

//   CHECK-NOT: memref.copy
  memref.copy %arg0, %arg0 : memref<2x3x?x4xf32> to memref<2x3x?x4xf32>

//   CHECK: return
  return
}

// -----
// CHECK-LABEL: func @fold_fill_reshape()
func.func @fold_fill_reshape() -> tensor<6x4xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x2x3x4xf32>
  // CHECK:      %[[COLLAPSE:.+]] = tensor.collapse_shape
  // CHECK-NEXT: %[[FILL:.+]] = linalg.fill ins(%cst : f32)
  // CHECK-SAME:   outs(%[[COLLAPSE]] : tensor<6x4xf32>)
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %reshape = tensor.collapse_shape %fill [[0, 1, 2], [3]]
      : tensor<1x2x3x4xf32> into tensor<6x4xf32>
  // CHECK: return %[[FILL]] : tensor<6x4xf32>
  return %reshape : tensor<6x4xf32>
}

// -----

//       CHECK: func @fold_fill_reshape_dynamic
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?x?x?xf32>
func.func @fold_fill_reshape_dynamic(%arg0 : tensor<?x?x?x?x?xf32>) -> tensor<?x?xf32> {
  %zero = arith.constant 0.0 : f32
  // CHECK: %[[RESHAPE:.+]] = tensor.collapse_shape %[[ARG0]]
  %0 = linalg.fill ins(%zero : f32) outs(%arg0 : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  // CHECK: %[[RESULT:.+]] = linalg.fill ins(%{{.+}}{{.*}}outs(%[[RESHAPE]]
  %1 = tensor.collapse_shape %0 [[0, 1, 2], [3, 4]]
      : tensor<?x?x?x?x?xf32> into tensor<?x?xf32>
  // CHECK: return %[[RESULT]]
  return %1 : tensor<?x?xf32>
}

// -----
//       CHECK: func @fold_fill_extract
//  CHECK-SAME:   %[[ARG0:.+]]: i1
func.func @fold_fill_extract(%arg0 : i1) -> i1 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %empty_dynamic = tensor.empty(%c1) : tensor<1x2x3x?xi1>
  %filled = linalg.fill ins(%arg0 : i1) outs(%empty_dynamic : tensor<1x2x3x?xi1>) -> tensor<1x2x3x?xi1>

  %extracted = tensor.extract %filled[%c0, %c0, %c0, %c0] : tensor<1x2x3x?xi1>

  //  CHECK:   return %[[ARG0]]
  return %extracted : i1
}

// -----

func.func @fill_pack() -> tensor<24x32x16x16xf32> {
  %dest = tensor.empty() : tensor<384x512xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<24x32x16x16xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%dest : tensor<384x512xf32>) -> tensor<384x512xf32>
  %pack = linalg.pack %1 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %0 : tensor<384x512xf32> -> tensor<24x32x16x16xf32>
  return %pack : tensor<24x32x16x16xf32>
}
// CHECK-LABEL: func.func @fill_pack
// CHECK:         %[[PACKED_EMPTY:.+]] = tensor.empty() : tensor<24x32x16x16xf32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%{{.+}}) outs(%[[PACKED_EMPTY]]
// CHECK:         return %[[FILL]]

// -----

func.func @fill_pack_general() -> tensor<1x1x8x4x4x8xi32>{
  %c0_i32 = arith.constant 0 : i32
  %alloc = memref.alloc() : memref<1x1x8x4x4x8xi32>
  %9 = tensor.empty() : tensor<1x1x16x64xi32>
  %extracted_slice_15 = tensor.extract_slice %9[0, 0, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : tensor<1x1x16x64xi32> to tensor<1x1x16x64xi32>
  %16 = linalg.fill ins(%c0_i32 : i32) outs(%extracted_slice_15 : tensor<1x1x16x64xi32>) -> tensor<1x1x16x64xi32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<1x1x8x4x4x8xi32> to tensor<1x1x8x4x4x8xi32>
  %pack_18 = linalg.pack %16 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %0 : tensor<1x1x16x64xi32> -> tensor<1x1x8x4x4x8xi32>
  return %pack_18 : tensor<1x1x8x4x4x8xi32>
}

// CHECK-LABEL: func.func @fill_pack_general
// CHECK:         %[[ALLOC:.+]] = memref.alloc() : memref<1x1x8x4x4x8xi32>
// CHECK:         %[[TENSOR:.+]] = bufferization.to_tensor %[[ALLOC]]
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%{{.+}}) outs(%[[TENSOR]]
// CHECK:         return %[[FILL]]

// -----

#map = affine_map<()[s0] -> (s0 ceildiv 16)>
func.func @dynamic_fill_pack(%arg0: tensor<?x?xf32>) -> tensor<?x?x16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %dim = tensor.dim %0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %0, %c1 : tensor<?x?xf32>
  %1 = affine.apply #map()[%dim]
  %2 = affine.apply #map()[%dim_0]
  %3 = tensor.empty(%1, %2) : tensor<?x?x16x16xf32>
  %pack = linalg.pack %0 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %3 : tensor<?x?xf32> -> tensor<?x?x16x16xf32>
  return %pack : tensor<?x?x16x16xf32>
}
// CHECK-DAG:   #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK:       func.func @dynamic_fill_pack
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK:         %[[D0:.+]] = tensor.dim %[[DEST]], %[[C0]]
// CHECK:         %[[D1:.+]] = tensor.dim %[[DEST]], %[[C1]]
// CHECK:         %[[PACKED_D0:.+]] = affine.apply #[[MAP]]()[%[[D0]]]
// CHECK:         %[[PACKED_D1:.+]] = affine.apply #[[MAP]]()[%[[D1]]]
// CHECK:         %[[PACKED_EMPTY:.+]] = tensor.empty(%[[PACKED_D0]], %[[PACKED_D1]]) : tensor<?x?x16x16xf32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%{{.+}}) outs(%[[PACKED_EMPTY]]
// CHECK:         return %[[FILL]]

// -----

// CHECK: func @fold_self_copy
func.func @fold_self_copy(%0 : memref<4x16xf32>) {
// CHECK-NEXT: return
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                   affine_map<(d0, d1) -> (d0, d1)>],
                  iterator_types = ["parallel", "parallel"]}
    ins(%0 : memref<4x16xf32>)
    outs(%0 : memref<4x16xf32>) {
      ^bb0(%arg4: f32, %arg5: f32):
        linalg.yield %arg4 : f32
    }
  return
}

// -----

// CHECK-LABEL: func @fold_static_pad_fill
//       CHECK:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<412x276xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[F0]]{{.*}}outs(%[[INIT]]
//       CHECK:   return %[[FILL]]
func.func @fold_static_pad_fill() -> tensor<412x276xf32> {
  %f0 = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<400x273xf32>
  %fill = linalg.fill ins(%f0 : f32) outs(%empty : tensor<400x273xf32>) -> tensor<400x273xf32>
  %pad = tensor.pad %fill low[4, 1] high[8, 2] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %f0 : f32
  } : tensor<400x273xf32> to tensor<412x276xf32>
  return %pad : tensor<412x276xf32>
}

// -----

// CHECK: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 + 9)>
// CHECK: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 + 10)>
// CHECK: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 + 23)>
// CHECK: #[[MAP3:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 32)>

//      CHECK: func @fold_dynamic_pad_fill
// CHECK-SAME: %[[SRC:.+]]: tensor<8x?x16x32xf32>, %[[LOW0:.+]]: index, %[[LOW3:.+]]: index, %[[HIGH2:.+]]: index, %[[HIGH3:.+]]: index

//  CHECK-DAG:   %[[I1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32
//      CHECK:   %[[S0:.+]] = affine.apply #[[MAP0]]()[%[[LOW0]]]
//      CHECK:   %[[DIM1:.+]] = tensor.dim %[[SRC]], %[[I1]] : tensor<8x?x16x32xf32>
//      CHECK:   %[[S1:.+]] = affine.apply #[[MAP1]]()[%[[DIM1]]]
//      CHECK:   %[[S2:.+]] = affine.apply #[[MAP2]]()[%[[HIGH2]]]
//      CHECK:   %[[S3:.+]] = affine.apply #[[MAP3]]()[%[[LOW3]], %[[HIGH3]]]
//      CHECK:   %[[INIT:.+]] = tensor.empty(%[[S0]], %[[S1]], %[[S2]], %[[S3]]) : tensor<?x?x?x?xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[F0]]{{.*}}outs(%[[INIT]]
//      CHECK:   return %[[FILL]]
func.func @fold_dynamic_pad_fill(%empty: tensor<8x?x16x32xf32>, %low0: index, %low3: index, %high2: index, %high3: index) -> tensor<?x?x?x?xf32> {
  %f0 = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%f0 : f32) outs(%empty : tensor<8x?x16x32xf32>) -> tensor<8x?x16x32xf32>
  %pad = tensor.pad %fill low[%low0, 8, 7, %low3] high[1, 2, %high2, %high3] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %f0 : f32
  } : tensor<8x?x16x32xf32> to tensor<?x?x?x?xf32>
  return %pad : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @no_fold_pad_fill_value_mismatch
func.func @no_fold_pad_fill_value_mismatch() -> tensor<412x276xf32> {
  %f0 = arith.constant 0.0 : f32
  %f1 = arith.constant 1.0 : f32
  %empty = tensor.empty() : tensor<400x273xf32>
  %fill = linalg.fill ins(%f0 : f32) outs(%empty : tensor<400x273xf32>) -> tensor<400x273xf32>
  // CHECK: tensor.pad
  %pad = tensor.pad %fill low[4, 1] high[8, 2] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %f1 : f32
  } : tensor<400x273xf32> to tensor<412x276xf32>
  return %pad : tensor<412x276xf32>
}

// -----

// Tests below verify whether static information is propagated through all the operands of generic op.
// 1. If one of the inputs of generic op has static info and it has no cast source.
// 2. If one of the inputs of generic op has static info and it is coming from tensr.cast operation.
// 3. If one of the outputs of generic op has static info and it is coming from tenso.cast operation.
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @static_input_without_cast
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x3x4xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>) -> tensor<2x3x4xf32> {
func.func @static_input_without_cast(%arg0 : tensor<2x3x4xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<2x3x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<2x3x4xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<2x3x4xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<2x3x4xf32>
  %3 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %4 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<2x3x4xf32>, tensor<?x?x?xf32>)
    outs(%3 : tensor<?x?x?xf32>) {
  ^bb0(%arg2 : f32, %arg3 : f32, %arg4 : f32):
    %9 = arith.addf %arg2, %arg3 : f32
    linalg.yield %9 : f32
  } -> (tensor<?x?x?xf32>)
  %5 = tensor.cast %4 : tensor<?x?x?xf32> to tensor<2x3x4xf32>
  return %5 : tensor<2x3x4xf32>
    //  CHECK:      %[[CAST_ARG1:.*]] = tensor.cast %[[ARG1]] : tensor<?x?x?xf32> to tensor<2x3x4xf32>
    //  CHECK-NEXT: %[[GENERIC_OP:.*]] = linalg.generic
    //  CHECK-SAME: ins(%[[ARG0]], %[[CAST_ARG1]] : tensor<2x3x4xf32>, tensor<2x3x4xf32>)
    //  CHECK-SAME: outs({{.*}} : tensor<2x3x4xf32>)
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @static_input_with_cast
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x3x4xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>) -> tensor<2x3x4xf32> {
func.func @static_input_with_cast(%arg0 : tensor<2x3x4xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<2x3x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<2x3x4xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<2x3x4xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<2x3x4xf32>
  %3 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %4 = tensor.cast %arg1 : tensor<?x?x?xf32> to tensor<2x?x?xf32>
  %5 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %4 : tensor<2x3x4xf32>, tensor<2x?x?xf32>)
    outs(%3 : tensor<?x?x?xf32>) {
  ^bb0(%arg2 : f32, %arg3 : f32, %arg4 : f32):
    %9 = arith.addf %arg2, %arg3 : f32
    linalg.yield %9 : f32
  } -> (tensor<?x?x?xf32>)
  %6 = tensor.cast %5 : tensor<?x?x?xf32> to tensor<2x3x4xf32>
  return %6: tensor<2x3x4xf32>
    //  CHECK:      %[[CAST_ARG1:.*]] = tensor.cast %[[ARG1]] : tensor<?x?x?xf32> to tensor<2x3x4xf32>
    //  CHECK-NEXT: %[[GENERIC_OP:.*]] = linalg.generic
    //  CHECK-SAME: ins(%[[ARG0]], %[[CAST_ARG1]] : tensor<2x3x4xf32>, tensor<2x3x4xf32>)
    //  CHECK-SAME: outs({{.*}} : tensor<2x3x4xf32>)
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @static_output_with_cast
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
func.func @static_output_with_cast(%arg0 : tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg2, %c0 : tensor<2x3x4xf32>
  %1 = tensor.dim %arg2, %c1 : tensor<2x3x4xf32>
  %2 = tensor.dim %arg2, %c2 : tensor<2x3x4xf32>
  %3 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %4 = tensor.cast %3 : tensor<?x?x?xf32> to tensor<2x3x4xf32>
  %5 = tensor.cast %arg1 : tensor<?x?x?xf32> to tensor<2x?x?xf32>
  %6 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %5 : tensor<?x?x?xf32>, tensor<2x?x?xf32>)
    outs(%4 : tensor<2x3x4xf32>) {
  ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
    %9 = arith.addf %arg3, %arg4 : f32
    linalg.yield %9 : f32
  } -> (tensor<2x3x4xf32>)
  return %6: tensor<2x3x4xf32>
    //  CHECK:      %[[CAST_ARG0:.*]] = tensor.cast %[[ARG0]] : tensor<?x?x?xf32> to tensor<2x3x4xf32>
    //  CHECK-NEXT: %[[CAST_ARG1:.*]] = tensor.cast %[[ARG1]] : tensor<?x?x?xf32> to tensor<2x3x4xf32>
    //  CHECK-NEXT: %[[GENERIC_OP:.*]] = linalg.generic
    //  CHECK-SAME: ins(%[[CAST_ARG0]], %[[CAST_ARG1]] : tensor<2x3x4xf32>, tensor<2x3x4xf32>)
    //  CHECK-SAME: outs({{.*}} : tensor<2x3x4xf32>)
}

// -----

// This test checks the folding of tensor.cast operation when the source value of cast
// has more static information than the destination value.
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @cast_source
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x3x4xf32>, %[[ARG1:.*]]: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
func.func @cast_source(%arg0 : tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<2x3x4xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<2x3x4xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<2x3x4xf32>
  %3 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %4 = tensor.cast %arg0 : tensor<2x3x4xf32> to tensor<2x?x?xf32>
  %5 = tensor.cast %arg1 : tensor<2x3x4xf32> to tensor<2x?x?xf32>
  %6 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%4, %5 : tensor<2x?x?xf32>, tensor<2x?x?xf32>)
    outs(%3 : tensor<?x?x?xf32>) {
  ^bb0(%arg2 : f32, %arg3 : f32, %arg4 : f32):
    %9 = arith.addf %arg2, %arg3 : f32
    linalg.yield %9 : f32
  } -> (tensor<?x?x?xf32>)
  %7 = tensor.cast %6 : tensor<?x?x?xf32> to tensor<2x3x4xf32>
  return %7: tensor<2x3x4xf32>
    //  CHECK:      %[[GENERIC_OP:.*]] = linalg.generic
    //  CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3x4xf32>, tensor<2x3x4xf32>)
    //  CHECK-SAME: outs({{.*}} : tensor<2x3x4xf32>)
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @cast_dest
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<1x?x?xf32>,
func.func @cast_dest(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x?x?xf32>, %arg2: index, %arg3: index, %arg4: index) -> tensor<?x?x?xf32> {
  %0 = tensor.empty(%arg2, %arg3, %arg4) : tensor<?x?x?xf32>
  %1 = tensor.cast %arg1 : tensor<1x?x?xf32> to tensor<?x?x?xf32>
  %2 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<1x?x?xf32>)
    outs(%0 : tensor<?x?x?xf32>) {
  ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):
    %3 = arith.subf %arg5, %arg6 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?x?xf32>
  return %2 : tensor<?x?x?xf32>
// CHECK:      %[[GENERIC_OP:.*]] = linalg.generic
// CHECK-SAME: ins(%{{.*}}, %[[ARG1]] : tensor<1x?x?xf32>, tensor<1x?x?xf32>)
// CHECK-SAME: outs(%{{.*}} : tensor<1x?x?xf32>)
// CHECK: tensor.cast %[[GENERIC_OP]] : tensor<1x?x?xf32> to tensor<?x?x?xf32>
}

// -----

//       CHECK: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 + 1)>
// CHECK-LABEL: func @insert_pad_into_fill
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<?x?x?xf32>, %[[LOW0:.+]]: index, %[[LOW1:.+]]: index, %{{.+}}: index, %{{.+}}: index)
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG: %[[F0:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK: %[[INIT:.+]] = tensor.empty()
//       CHECK: %[[FILL:.+]] = linalg.fill ins(%[[F0]]{{.*}}outs(%[[INIT]]
//       CHECK: %[[OFFSET1:.+]] = affine.apply #[[$MAP]]()[%[[LOW1]]]
//       CHECK: %[[D0:.+]] = tensor.dim %[[INPUT]], %[[C0]] : tensor<?x?x?xf32>
//       CHECK: %[[D1:.+]] = tensor.dim %[[INPUT]], %[[C1]] : tensor<?x?x?xf32>
//       CHECK: %[[D2:.+]] = tensor.dim %[[INPUT]], %[[C2]] : tensor<?x?x?xf32>
//       CHECK: tensor.insert_slice %[[INPUT]] into %[[FILL]][%[[LOW0]], %[[OFFSET1]], 2] [%[[D0]], %[[D1]], %[[D2]]] [1, 1, 1]
func.func @insert_pad_into_fill(%input: tensor<?x?x?xf32>, %low0: index, %low1: index, %high1: index, %high2: index) -> tensor<8x384x384xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %pad = tensor.pad %input low[%low0, %low1, %c0] high[%c0, %high1, %high2] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %f0 : f32
  } : tensor<?x?x?xf32> to tensor<8x128x128xf32>
  %empty = tensor.empty() : tensor<8x384x384xf32>
  %fill = linalg.fill ins(%f0 : f32) outs(%empty : tensor<8x384x384xf32>) -> tensor<8x384x384xf32>
  %0 = tensor.insert_slice %pad into %fill[0, 1, 2] [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  return %0: tensor<8x384x384xf32>
}

// -----

// CHECK-LABEL: func @multi_insert_pad_into_fill
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<7x123x124xf32>, %[[A:.+]]: tensor<8x128x128xf32>, %[[OFFSET:.+]]: index)
//       CHECK:   %[[FILL:.+]] = linalg.fill
//       CHECK:   %[[INSERT0:.+]] = tensor.insert_slice %[[A]] into %[[FILL]][%[[OFFSET]], 0, 0] [8, 128, 128] [1, 1, 1]
//       CHECK:   %[[INSERT1:.+]] = tensor.insert_slice %[[A]] into %[[INSERT0]][0, 128, %[[OFFSET]]] [8, 128, 128] [1, 1, 1]
//       CHECK:                  tensor.insert_slice %[[INPUT]] into %[[INSERT1]][1, 2, 256] [7, 123, 124] [1, 1, 1]
func.func @multi_insert_pad_into_fill(%input: tensor<7x123x124xf32>, %a: tensor<8x128x128xf32>, %offset: index) -> tensor<8x384x384xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %pad = tensor.pad %input low[1, 2, 0] high[0, 3, 4] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %f0 : f32
  } : tensor<7x123x124xf32> to tensor<8x128x128xf32>
  %empty = tensor.empty() : tensor<8x384x384xf32>
  %fill = linalg.fill ins(%f0 : f32) outs(%empty : tensor<8x384x384xf32>) -> tensor<8x384x384xf32>
  %0 = tensor.insert_slice %a   into %fill[%offset, 0, 0]  [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %1 = tensor.insert_slice %a   into %0   [0, 128, %offset][8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %2 = tensor.insert_slice %pad into %1   [0, 0, 256]      [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  return %2: tensor<8x384x384xf32>
}

// -----

// CHECK-LABEL: func @multi_insert_pad_into_fill_overlap
func.func @multi_insert_pad_into_fill_overlap(%input: tensor<7x123x124xf32>, %a: tensor<8x128x128xf32>, %offset: index) -> tensor<8x384x384xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK: tensor.pad
  %pad = tensor.pad %input low[1, 2, 0] high[0, 3, 4] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %f0 : f32
  } : tensor<7x123x124xf32> to tensor<8x128x128xf32>
  %empty = tensor.empty() : tensor<8x384x384xf32>
  %fill = linalg.fill ins(%f0 : f32) outs(%empty : tensor<8x384x384xf32>) -> tensor<8x384x384xf32>
  %0 = tensor.insert_slice %a   into %fill[%offset, 0, 0]  [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %1 = tensor.insert_slice %a   into %0   [0, 0, 129]      [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  // Range overlap with %1 at dim#3
  %2 = tensor.insert_slice %pad into %1   [0, 0, 256]      [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  return %2: tensor<8x384x384xf32>
}

// -----

// CHECK-LABEL: func @multi_insert_pad_into_fill_overlap
func.func @multi_insert_pad_into_fill_overlap(%input: tensor<7x123x124xf32>, %a: tensor<8x128x128xf32>, %offset: index) -> tensor<8x384x384xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK: tensor.pad
  %pad = tensor.pad %input low[1, 2, 0] high[0, 3, 4] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %f0 : f32
  } : tensor<7x123x124xf32> to tensor<8x128x128xf32>
  %empty = tensor.empty() : tensor<8x384x384xf32>
  %fill = linalg.fill ins(%f0 : f32) outs(%empty : tensor<8x384x384xf32>) -> tensor<8x384x384xf32>
  %0 = tensor.insert_slice %a   into %fill[0, 0, %offset]  [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %1 = tensor.insert_slice %a   into %0   [0, 128, 255]    [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  // Range overlap with %0 at dim#3
  %2 = tensor.insert_slice %pad into %1   [0, 0, 256]      [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  return %2: tensor<8x384x384xf32>
}

// -----

// CHECK-LABEL: func @multi_insert_pad_into_fill
func.func @multi_insert_pad_into_fill(%input: tensor<7x123x124xf32>, %a: tensor<8x128x128xf32>, %offset: index) -> tensor<8x384x384xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK-NOT: tensor.pad
  %pad = tensor.pad %input low[1, 2, 0] high[0, 3, 4] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %f0 : f32
  } : tensor<7x123x124xf32> to tensor<8x128x128xf32>
  %empty = tensor.empty() : tensor<8x384x384xf32>
  %fill = linalg.fill ins(%f0 : f32) outs(%empty : tensor<8x384x384xf32>) -> tensor<8x384x384xf32>
  // Overlap btween %0 and %1 is fine but not with %2 is fine.
  // CHECK-COUNT-3: tensor.insert_slice
  %0 = tensor.insert_slice %a   into %fill[0, 0, %offset]  [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %1 = tensor.insert_slice %a   into %0   [0, 1, %offset]  [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %2 = tensor.insert_slice %pad into %1   [0, 256, 256]    [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  return %2: tensor<8x384x384xf32>
}

// -----

// CHECK-LABEL: func @multi_insert_pad_into_fill_mismatch
func.func @multi_insert_pad_into_fill_mismatch(%input: tensor<7x123x124xf32>, %a: tensor<8x128x128xf32>, %offset: index) -> tensor<8x384x384xf32> {
  %f0 = arith.constant 0.0 : f32
  %f1 = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK: tensor.pad
  %pad = tensor.pad %input low[1, 2, 0] high[0, 3, 4] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %f0 : f32
  } : tensor<7x123x124xf32> to tensor<8x128x128xf32>
  %empty = tensor.empty() : tensor<8x384x384xf32>
  // Different filling value than padding value.
  %fill = linalg.fill ins(%f1 : f32) outs(%empty : tensor<8x384x384xf32>) -> tensor<8x384x384xf32>
  %0 = tensor.insert_slice %a   into %fill[%offset, 0, 0]  [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %1 = tensor.insert_slice %a   into %0   [0, 128, %offset][8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %2 = tensor.insert_slice %pad into %1   [0, 0, 256]      [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  return %2: tensor<8x384x384xf32>
}

// -----

func.func @fold_linalgop_with_cast_consumer(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> (tensor<4x8xf32>, tensor<?x?xf32>) {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = tensor.cast %0 : tensor<?x?xf32> to tensor<4x8xf32>
  return %1, %0 : tensor<4x8xf32>, tensor<?x?xf32>
}
//       CHECK: func @fold_linalgop_with_cast_consumer(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//   CHECK-DAG:  %[[LHS_CAST:.+]] = tensor.cast %[[ARG0]] : tensor<?x?xf32> to tensor<4x?xf32>
//   CHECK-DAG:  %[[RHS_CAST:.+]] = tensor.cast %[[ARG1]] : tensor<?x?xf32> to tensor<?x8xf32>
//   CHECK-DAG:  %[[OUT_CAST:.+]] = tensor.cast %[[ARG2]] : tensor<?x?xf32> to tensor<4x8xf32>
//       CHECK:  %[[MATMUL:.+]] = linalg.matmul
//  CHECK-SAME:      ins(%[[LHS_CAST]], %[[RHS_CAST]] :
//  CHECK-SAME:      outs(%[[OUT_CAST]] :
//       CHECK:  %[[RESULT_CAST:.+]] = tensor.cast %[[MATMUL]]
//       CHECK:  return %[[MATMUL]], %[[RESULT_CAST]]

// -----

func.func private @some_use(%0 : tensor<4x8xf32>)

func.func @linalgop_with_cond_cast_consumer(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>, %arg3 : i1) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  scf.if %arg3 {
    %1 = tensor.cast %0 : tensor<?x?xf32> to tensor<4x8xf32>
    func.call @some_use(%1) : (tensor<4x8xf32>) -> ()
  }
  return %0 : tensor<?x?xf32>
}

// Check conditionally reachable cast is not folded into producer.
// CHECK-LABEL: func @linalgop_with_cond_cast_consumer
//  CHECK-SAME:     (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>, %[[ARG3:.*]]: i1)
//       CHECK: %[[RES:.*]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>)
//  CHECK-SAME:      outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
//       CHECK: scf.if %[[ARG3]] {
//       CHECK:   %[[CAST:.*]] = tensor.cast %[[RES]] : tensor<?x?xf32> to tensor<4x8xf32>
//       CHECK:   func.call @some_use(%[[CAST]]) : (tensor<4x8xf32>) -> ()
//       CHECK: }
//       CHECK: return %[[RES]] : tensor<?x?xf32>


// -----

func.func @fold_conv_op_with_cast_consumer(%arg0 : tensor<?x?x?x?xf32>,
    %arg1 : tensor<?x?x?x?xf32>,  %arg2 : tensor<?x?x?x?xf32>) ->
    (tensor<4x8x12x16xf32>, tensor<?x?x?x?xf32>) {
  %0 = linalg.conv_2d_nchw_fchw ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1 = tensor.cast %0 : tensor<?x?x?x?xf32> to tensor<4x8x12x16xf32>
  return %1, %0 : tensor<4x8x12x16xf32>, tensor<?x?x?x?xf32>
}
//       CHECK: func @fold_conv_op_with_cast_consumer(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>)
//       CHECK:  %[[OUT_CAST:.+]] = tensor.cast %[[ARG2]] : tensor<?x?x?x?xf32> to tensor<4x8x12x16xf32>
//       CHECK:  %[[CONV:.+]] = linalg.conv_2d_nchw_fchw
//  CHECK-SAME:      ins(%[[ARG0]], %[[ARG1]] :
//  CHECK-SAME:      outs(%[[OUT_CAST]] :
//       CHECK:  %[[RESULT_CAST:.+]] = tensor.cast %[[CONV]]
//       CHECK:  return %[[CONV]], %[[RESULT_CAST]]

// -----

func.func @fold_multi_use_generic_op_with_consumer(%arg0 : tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<2x3x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %empty1 = tensor.empty(%d1, %d2, %d0) : tensor<?x?x?xf32>
  %empty2 = tensor.empty(%d2, %d1, %d0) : tensor<?x?x?xf32>
  %0:2 = linalg.generic {
      iterator_types = ["parallel", "parallel", "parallel"],
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d2, d1, d0)>]}
      ins(%arg0 : tensor<?x?x?xf32>) outs(%empty1, %empty2 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32) :
      linalg.yield %b0, %b0 : f32, f32
    } -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  %1 = tensor.cast %0#1 : tensor<?x?x?xf32> to tensor<2x3x4xf32>
  return %0#0, %1 : tensor<?x?x?xf32>, tensor<2x3x4xf32>
}
//       CHECK: func @fold_multi_use_generic_op_with_consumer
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>
//   CHECK-DAG:   %[[INIT1:.+]] = tensor.empty() : tensor<2x3x4xf32>
//   CHECK-DAG:   %[[CAST:.+]] = tensor.cast %[[ARG0]] : tensor<?x?x?xf32> to tensor<4x3x2xf32>
//   CHECK-DAG:   %[[INIT2:.+]] = tensor.empty() : tensor<3x2x4xf32>
//       CHECK:   %[[GENERIC:.+]]:2 = linalg.generic
//  CHECK-SAME:       ins(%[[CAST]] :
//  CHECK-SAME:       outs(%[[INIT2]], %[[INIT1]] :
//       CHECK:   %[[RETURN_CAST:.+]] = tensor.cast %[[GENERIC]]#0 : tensor<3x2x4xf32> to tensor<?x?x?xf32>
//       CHECK:   return %[[RETURN_CAST]], %[[GENERIC]]#1

// -----

#map = affine_map<(d0) -> (d0)>
func.func @identity_buffer(%arg0 : memref<?xf32>, %arg1: memref<?xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]
  } ins(%arg0 : memref<?xf32>)
    outs(%arg1 : memref<?xf32>) {
  ^bb0(%arg2 : f32, %arg3 : f32):
    linalg.yield %arg2 : f32
  }
  return
}

// Do not erase ops with buffer semantics.
// CHECK-LABEL: func @identity_buffer
//  CHECK-SAME:     (%[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: memref<?xf32>)
//       CHECK:     linalg.generic {
//  CHECK-SAME:    indexing_maps = [#map, #map],
//  CHECK-SAME:    iterator_types = ["parallel"]
//  CHECK-SAME:  } ins(%[[ARG1]] : memref<?xf32>)
//  CHECK-SAME:    outs(%[[ARG2]] : memref<?xf32>) {

// -----

#map = affine_map<(d0, d1) -> (d1, d0)>
func.func @erase_non_identity_noop(%arg0 : tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<?x?xf32>)
    outs(%arg1 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in: f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// Do not erase ops with buffer semantics.
// CHECK-LABEL: func @erase_non_identity_noop
//  CHECK-SAME:   (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>)
//       CHECK:   return %[[ARG0]] : tensor<?x?xf32>

// -----

// Just make sure that we don't crash.

// CHECK-LABEL: func @dedeplicate_regression_test
func.func @dedeplicate_regression_test(%0: tensor<4xf32>, %1: tensor<4xf32>) {
  %36 = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0)>,
                      affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]}
    ins(%1, %1 : tensor<4xf32>, tensor<4xf32>)
    outs(%0 : tensor<4xf32>) {
  ^bb0(%in: f32, %in_24: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4xf32>
  %53 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>],
                        iterator_types = ["parallel"]}
                        outs(%36 : tensor<4xf32>) {
  ^bb0(%out: f32):
    linalg.yield %out : f32
  } -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: dead_softmax
func.func @dead_softmax(%arg0: tensor<16x64x256xf32>) -> tensor<16x64x256xf32> {
  %0 = tensor.empty() : tensor<16x64x256xf32>
  // CHECK-NOT: linalg.softmax
  %1 = linalg.softmax dimension(1)
    ins(%arg0 : tensor<16x64x256xf32>) outs(%0 : tensor<16x64x256xf32>) -> tensor<16x64x256xf32>
  return %arg0 : tensor<16x64x256xf32>
}

// -----

// CHECK-LABEL: func @canonicalize_dim_of_dest_style_op
//       CHECK: tensor.dim
//       CHECK: tensor.dim
//   CHECK-NOT: tensor.dim
//       CHECK: return
func.func @canonicalize_dim_of_dest_style_op(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0_0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim1_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim0_0, %dim1_0) : tensor<?x?xf32>
  %1 = linalg.copy ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %dim0_1 = tensor.dim %1, %c0 : tensor<?x?xf32>
  %dim1_1 = tensor.dim %1, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%dim0_1, %dim1_1) : tensor<?x?xf32>
  %3 = linalg.copy ins(%1 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %3: tensor<?x?xf32>
}
// -----

// CHECK-LABEL: func @canonicalize_fill_to_copy_input(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//       CHECK:   %[[ZERO:.+]] = arith.constant 0.0
//       CHECK:   linalg.fill ins(%[[ZERO]] : f32) outs(%[[ARG1]] : tensor<?x?xf32>)
func.func @canonicalize_fill_to_copy_input(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%c0 : f32) outs(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %copy = linalg.copy ins(%fill : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %copy : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @canonicalize_fill_to_copy_dest(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//       CHECK:   linalg.copy ins(%[[ARG1]] : tensor<?x?xf32>) outs(%[[ARG0]] : tensor<?x?xf32>)
func.func @canonicalize_fill_to_copy_dest(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%c0 : f32) outs(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %copy = linalg.copy ins(%arg1 : tensor<?x?xf32>) outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %copy : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @canonicalize_fill_to_transpose_input(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//       CHECK:   %[[ZERO:.+]] = arith.constant 0.0
//       CHECK:   linalg.fill ins(%[[ZERO]] : f32) outs(%[[ARG1]] : tensor<?x?xf32>)
func.func @canonicalize_fill_to_transpose_input(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%c0 : f32) outs(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %transpose = linalg.transpose ins(%fill : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) permutation = [1, 0]
  return %transpose : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @broadcast_same_shape(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x3xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<2x3xf32>)
//       CHECK-NOT:   linalg.broadcast
//       CHECK:       return %[[ARG0]] : tensor<2x3xf32>
func.func @broadcast_same_shape(%input: tensor<2x3xf32>, %init: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = linalg.broadcast ins(%input: tensor<2x3xf32>) outs(%init: tensor<2x3xf32>) dimensions = []
  return %0 : tensor<2x3xf32>
}

// -----

func.func @transpose_1d(%input: tensor<16xf32>,
                        %init: tensor<16xf32>) -> tensor<16xf32> {
  %transpose = linalg.transpose
      ins(%input:tensor<16xf32>)
      outs(%init:tensor<16xf32>)
      permutation = [0]
  func.return %transpose : tensor<16xf32>
}

// CHECK-LABEL: func @transpose_1d(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<16xf32>,
//  CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: tensor<16xf32>)
//   CHECK-NOT:   linalg.transpose
//       CHECK:   return %[[INPUT]] : tensor<16xf32>

// -----

func.func @transpose_identity_perm(%input: tensor<16x32x64xf32>,
                                   %init: tensor<16x32x64xf32>) -> tensor<16x32x64xf32> {
  %transpose = linalg.transpose
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x32x64xf32>)
      permutation = [0, 1, 2]
  func.return %transpose : tensor<16x32x64xf32>
}

// CHECK-LABEL: func @transpose_identity_perm(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<16x32x64xf32>,
//  CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: tensor<16x32x64xf32>)
//   CHECK-NOT:   linalg.transpose
//       CHECK:   return %[[INPUT]] : tensor<16x32x64xf32>

// -----

func.func @transpose_transpose_cancel(%input: tensor<5x4x3xf32>,
                                      %init1: tensor<4x3x5xf32>,
                                      %init2: tensor<5x4x3xf32>) -> tensor<5x4x3xf32> {
  // CHECK-LABEL: @transpose_transpose_cancel
  //  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<5x4x3xf32>
  //  CHECK-SAME:     %[[INIT1:[a-zA-Z0-9]+]]: tensor<4x3x5xf32>
  //  CHECK-SAME:     %[[INIT2:[a-zA-Z0-9]+]]: tensor<5x4x3xf32>
  //   CHECK-NOT:   linalg.transpose
  //       CHECK:   return %[[INPUT]] : tensor<5x4x3xf32>
  %transpose1 = linalg.transpose
      ins(%input:tensor<5x4x3xf32>)
      outs(%init1:tensor<4x3x5xf32>)
      permutation = [1, 2, 0]
  %transpose2 = linalg.transpose
      ins(%transpose1:tensor<4x3x5xf32>)
      outs(%init2:tensor<5x4x3xf32>)
      permutation = [2, 0, 1]
  func.return %transpose2 : tensor<5x4x3xf32>
}

// -----

func.func @transpose_transpose_fold(%input: tensor<5x4x3xf32>,
                                    %init1: tensor<4x3x5xf32>,
                                    %init2: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
  // CHECK-LABEL: @transpose_transpose_fold
  //  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<5x4x3xf32>
  //  CHECK-SAME:     %[[INIT1:[a-zA-Z0-9]+]]: tensor<4x3x5xf32>
  //  CHECK-SAME:     %[[INIT2:[a-zA-Z0-9]+]]: tensor<3x4x5xf32>
  //       CHECK:   %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[INPUT]] : tensor<5x4x3xf32>) outs(%[[INIT2]] : tensor<3x4x5xf32>) permutation = [2, 1, 0]
  //   CHECK-NOT:   linalg.transpose
  //       CHECK:   return %[[TRANSPOSE]] : tensor<3x4x5xf32>
  %transpose1 = linalg.transpose
      ins(%input:tensor<5x4x3xf32>)
      outs(%init1:tensor<4x3x5xf32>)
      permutation = [1, 2, 0]
  %transpose2 = linalg.transpose
      ins(%transpose1:tensor<4x3x5xf32>)
      outs(%init2:tensor<3x4x5xf32>)
      permutation = [1, 0, 2]
  func.return %transpose2 : tensor<3x4x5xf32>
}

// -----

func.func @broadcast_transpose_fold(%input: tensor<2x4x5xf32>,
                                    %init1: tensor<1x2x3x4x5x6xf32>,
                                    %init2: tensor<1x6x2x3x5x4xf32>) -> tensor<1x6x2x3x5x4xf32> {
  // CHECK-LABEL: @broadcast_transpose_fold
  //  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<2x4x5xf32>
  //  CHECK-SAME:     %[[INIT1:[a-zA-Z0-9]+]]: tensor<1x2x3x4x5x6xf32>
  //  CHECK-SAME:     %[[INIT2:[a-zA-Z0-9]+]]: tensor<1x6x2x3x5x4xf32>
  //       CHECK:   %[[TMP_INIT:.+]] = tensor.empty() : tensor<2x5x4xf32>
  //       CHECK:   %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[INPUT]] : tensor<2x4x5xf32>) outs(%[[TMP_INIT]] : tensor<2x5x4xf32>) permutation = [0, 2, 1]
  //       CHECK:   %[[BROADCAST:.+]] = linalg.broadcast ins(%[[TRANSPOSE]] : tensor<2x5x4xf32>) outs(%[[INIT2]] : tensor<1x6x2x3x5x4xf32>) dimensions = [0, 3, 1]
  //       CHECK:   return %[[BROADCAST]] : tensor<1x6x2x3x5x4xf32>
  %broadcast = linalg.broadcast
      ins(%input : tensor<2x4x5xf32>)
      outs(%init1 : tensor<1x2x3x4x5x6xf32>)
      dimensions = [0, 2, 5]
  %transpose = linalg.transpose
      ins(%broadcast : tensor<1x2x3x4x5x6xf32>)
      outs(%init2 : tensor<1x6x2x3x5x4xf32>)
      permutation = [0, 5, 1, 2, 4, 3]
  func.return %transpose : tensor<1x6x2x3x5x4xf32>
}

// -----

func.func @broadcast_transpose_fold_dynamic(%input: tensor<?x?x5xf32>,
                                            %init1: tensor<1x?x3x?x5x6xf32>,
                                            %init2: tensor<1x3x?x6x5x?xf32>) -> tensor<1x3x?x6x5x?xf32> {
  // CHECK-LABEL: @broadcast_transpose_fold_dynamic
  //  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<?x?x5xf32>
  //  CHECK-SAME:     %[[INIT1:[a-zA-Z0-9]+]]: tensor<1x?x3x?x5x6xf32>
  //  CHECK-SAME:     %[[INIT2:[a-zA-Z0-9]+]]: tensor<1x3x?x6x5x?xf32>
  //   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
  //   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
  //       CHECK:   %[[DIM0:.+]] = tensor.dim %[[INPUT]], %[[C0]] : tensor<?x?x5xf32>
  //       CHECK:   %[[DIM1:.+]] = tensor.dim %[[INPUT]], %[[C1]] : tensor<?x?x5xf32>
  //       CHECK:   %[[TMP_INIT:.+]] = tensor.empty(%[[DIM1]], %[[DIM0]]) : tensor<?x5x?xf32>
  //       CHECK:   %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[INPUT]] : tensor<?x?x5xf32>) outs(%[[TMP_INIT]] : tensor<?x5x?xf32>) permutation = [1, 2, 0]
  //       CHECK:   %[[BROADCAST:.+]] = linalg.broadcast ins(%[[TRANSPOSE]] : tensor<?x5x?xf32>) outs(%[[INIT2]] : tensor<1x3x?x6x5x?xf32>) dimensions = [0, 1, 3]
  //       CHECK:   return %[[BROADCAST]] : tensor<1x3x?x6x5x?xf32>
  %broadcast = linalg.broadcast
      ins(%input : tensor<?x?x5xf32>)
      outs(%init1 : tensor<1x?x3x?x5x6xf32>)
      dimensions = [0, 2, 5]
  %transpose = linalg.transpose
      ins(%broadcast : tensor<1x?x3x?x5x6xf32>)
      outs(%init2 : tensor<1x3x?x6x5x?xf32>)
      permutation = [0, 2, 3, 5, 4, 1]
  func.return %transpose : tensor<1x3x?x6x5x?xf32>
}

// -----

func.func @broadcast_transpose_fold_2dim(%input: tensor<2xf32>,
                                         %init1: tensor<2x4xf32>,
                                         %init2: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // CHECK-LABEL: @broadcast_transpose_fold_2dim
  //  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<2xf32>
  //  CHECK-SAME:     %[[INIT1:[a-zA-Z0-9]+]]: tensor<2x4xf32>
  //  CHECK-SAME:     %[[INIT2:[a-zA-Z0-9]+]]: tensor<4x2xf32>
  //       CHECK:   %[[BROADCAST:.+]] = linalg.broadcast ins(%[[INPUT]] : tensor<2xf32>) outs(%[[INIT2]] : tensor<4x2xf32>) dimensions = [0]
  //       CHECK:   return %[[BROADCAST]] : tensor<4x2xf32>
  %broadcast = linalg.broadcast
      ins(%input : tensor<2xf32>)
      outs(%init1 : tensor<2x4xf32>)
      dimensions = [1]
  %transpose = linalg.transpose
      ins(%broadcast : tensor<2x4xf32>)
      outs(%init2 : tensor<4x2xf32>)
      permutation = [1, 0]
  func.return %transpose : tensor<4x2xf32>
}

// -----

func.func @concats_of_fill(
    %arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index)
    -> tensor<5x?x?xf32>
{
  %cst0 = arith.constant 0.0 : f32
  %cst1 = arith.constant 0.0 : f32
  %0 = tensor.empty(%arg0, %arg1) : tensor<5x?x?xf32>
  %1 = linalg.fill ins(%cst0 : f32) outs(%0 : tensor<5x?x?xf32>) -> tensor<5x?x?xf32>
  %2 = tensor.empty(%arg2, %arg3) : tensor<5x?x?xf32>
  %3 = linalg.fill ins(%cst1 : f32) outs(%2 : tensor<5x?x?xf32>) -> tensor<5x?x?xf32>
  %4 = tensor.concat dim(1) %1, %3 : (tensor<5x?x?xf32>, tensor<5x?x?xf32>) -> tensor<5x?x?xf32>
  return %4 : tensor<5x?x?xf32>
}
//       CHECK: func @concats_of_fill(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index)
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.0
//   CHECK-DAG:   %[[EMPTY0:.+]] = tensor.empty(%[[ARG0]], %[[ARG1]])
//   CHECK-DAG:   %[[EMPTY1:.+]] = tensor.empty(%[[ARG2]], %[[ARG3]])
//       CHECK:   %[[CONCAT:.+]] = tensor.concat dim(1) %[[EMPTY0]], %[[EMPTY1]]
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[CONCAT]] :
//       CHECK:   return %[[FILL]]

// -----

func.func @transpose_buffer(%input: memref<?xf32>,
                            %init: memref<?xf32>) {
  linalg.transpose ins(%input:memref<?xf32>)
                   outs(%init:memref<?xf32>)
                   permutation = [0]
  func.return
}

// CHECK-LABEL:   func.func @transpose_buffer(
//  CHECK-SAME:            %[[VAL_0:.*]]: memref<?xf32>,
//  CHECK-SAME:            %[[VAL_1:.*]]: memref<?xf32>) {
//       CHECK:     linalg.transpose ins(%[[VAL_0]] : memref<?xf32>)
//  CHECK-SAME:       outs(%[[VAL_1]] : memref<?xf32>) permutation = [0]

// -----

// This test checks linalg op has a recursive memory effect. Otherwise
// linalg.map without a user would be DCEd.
func.func @recursive_effect(%arg : tensor<1xf32>) {
  %init = arith.constant dense<0.0> : tensor<1xf32>
  %mapped = linalg.map ins(%arg:tensor<1xf32>) outs(%init :tensor<1xf32>)
            (%in : f32) {
              vector.print %in : f32
              linalg.yield %in : f32
            }
  func.return
}

// CHECK-LABEL: @recursive_effect
//       CHECK: linalg.map

//===----------------------------------------------------------------------===//
// linalg.pack
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @fold_pack_constant_splat
//   CHECK-NOT: linalg.pack
//       CHECK: arith.constant dense<1.000000e-01> : tensor<8x16x8x32xf32>
func.func @fold_pack_constant_splat(%dest : tensor<8x16x8x32xf32>) -> tensor<8x16x8x32xf32> {
  %cst = arith.constant dense<1.000000e-01> : tensor<64x128xf32>
  %0 = linalg.pack %cst outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
    inner_tiles = [8, 32] into %dest : tensor<64x128xf32> -> tensor<8x16x8x32xf32>
  return %0 : tensor<8x16x8x32xf32>
}

// -----

// CHECK-LABEL: func @fold_padding_value_pack_constant_splat
//   CHECK-NOT: linalg.pack
//       CHECK: arith.constant dense<1.000000e-01> : tensor<8x16x8x32xf32>
func.func @fold_padding_value_pack_constant_splat(%dest : tensor<8x16x8x32xf32>) -> tensor<8x16x8x32xf32> {
  %pad = arith.constant 1.000000e-01 : f32
  %cst = arith.constant dense<1.000000e-01> : tensor<63x127xf32>
  %0 = linalg.pack %cst
    padding_value(%pad : f32)
    outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
    inner_tiles = [8, 32] into %dest : tensor<63x127xf32> -> tensor<8x16x8x32xf32>
  return %0 : tensor<8x16x8x32xf32>
}


// -----

// CHECK-LABEL: func @nofold_padding_value_pack_constant_splat
//       CHECK: arith.constant dense<1.000000e-01> : tensor<63x127xf32>
//       CHECK: linalg.pack
func.func @nofold_padding_value_pack_constant_splat(%dest : tensor<8x16x8x32xf32>) -> tensor<8x16x8x32xf32> {
  %pad = arith.constant 0.0 : f32
  %cst = arith.constant dense<1.000000e-01> : tensor<63x127xf32>
  %0 = linalg.pack %cst
    padding_value(%pad : f32)
    outer_dims_perm = [1, 0]
    inner_dims_pos = [0, 1]
    inner_tiles = [8, 32]
    into %dest : tensor<63x127xf32> -> tensor<8x16x8x32xf32>
  return %0 : tensor<8x16x8x32xf32>
}

// -----

func.func @fold_padding_value_pack(%arg0: tensor<1200x500000xf32>) -> tensor<31250x1200x16x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<31250x1200x16x1xf32>
  %pack = linalg.pack %arg0
    padding_value(%cst : f32)
    outer_dims_perm = [1, 0]
    inner_dims_pos = [1, 0]
    inner_tiles = [16, 1]
    into %0 : tensor<1200x500000xf32> -> tensor<31250x1200x16x1xf32>
  return %pack : tensor<31250x1200x16x1xf32>
}
// CHECK-LABEL: func @fold_padding_value_pack
// CHECK-NOT:     padding_value

// -----

func.func @infer_src_shape_pack(%src: tensor<?x?x?x?xf32>, %dest: tensor<10x20x30x40x16xf32>) -> tensor<10x20x30x40x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
   %pack = linalg.pack %src
    padding_value(%cst : f32)
    outer_dims_perm = [2, 1, 3, 0]
    inner_dims_pos = [2]
    inner_tiles = [16]
    into %dest : tensor<?x?x?x?xf32> -> tensor<10x20x30x40x16xf32>
  return %pack : tensor<10x20x30x40x16xf32>
}
// CHECK-LABEL: func.func @infer_src_shape_pack
// CHECK-SAME:    %[[SRC:[0-9a-zA-Z]+]]
// CHECK-SAME:    %[[DEST:[0-9a-zA-Z]+]]
// CHECK:         %[[CAST_SRC:.+]] = tensor.cast %[[SRC]] : tensor<?x?x?x?xf32> to tensor<40x20x?x30xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[CAST_SRC]] {{.+}} into %[[DEST]]
// CHECK:         return %[[PACK]]

// -----

func.func @infer_dest_shape_pack(%src: tensor<30x20x?x10xf32>, %dest: tensor<?x?x?x?x16xf32>) -> tensor<?x?x?x?x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
   %pack = linalg.pack %src
    padding_value(%cst : f32)
    outer_dims_perm = [2, 1, 3, 0]
    inner_dims_pos = [2]
    inner_tiles = [16]
    into %dest : tensor<30x20x?x10xf32> -> tensor<?x?x?x?x16xf32>
  return %pack : tensor<?x?x?x?x16xf32>
}
// CHECK-LABEL: func.func @infer_dest_shape_pack
// CHECK-SAME:    %[[SRC:[0-9a-zA-Z]+]]
// CHECK-SAME:    %[[DEST:[0-9a-zA-Z]+]]
// CHECK:         %[[CAST_DEST:.+]] = tensor.cast %[[DEST]] : tensor<?x?x?x?x16xf32> to tensor<?x20x10x30x16xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[SRC]] {{.+}} into %[[CAST_DEST]]
// CHECK:         %[[CAST_PACK:.+]] = tensor.cast %[[PACK]] : tensor<?x20x10x30x16xf32> to tensor<?x?x?x?x16xf32>
// CHECK:         return %[[CAST_PACK]]

// -----

func.func @no_infer_pack_shape(%arg0: tensor<?x32x100xf32>, %arg1: index) -> tensor<32x7x?x16x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty(%arg1) : tensor<32x7x?x16x1xf32>
  %pack = linalg.pack %arg0 padding_value(%cst : f32) outer_dims_perm = [1, 2, 0] inner_dims_pos = [2, 0] inner_tiles = [16, 1] into %0 : tensor<?x32x100xf32> -> tensor<32x7x?x16x1xf32>
  return %pack : tensor<32x7x?x16x1xf32>
}
// CHECK-LABEL: func.func @no_infer_pack_shape
// CHECK-NOT:     tensor.cast

// -----

func.func @fold_padding_value_pack_negative1(%arg0: tensor<1200x499999xf32>) -> tensor<31250x1200x16x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<31250x1200x16x1xf32>
  %pack = linalg.pack %arg0
    padding_value(%cst : f32)
    outer_dims_perm = [1, 0]
    inner_dims_pos = [1, 0]
    inner_tiles = [16, 1]
    into %0 : tensor<1200x499999xf32> -> tensor<31250x1200x16x1xf32>
  return %pack : tensor<31250x1200x16x1xf32>
}
// CHECK-LABEL: func @fold_padding_value_pack_negative1
// CHECK:         linalg.pack
// CHECK-SAME:      padding_value

// -----

func.func @fold_padding_value_pack_negative2(%arg0: tensor<1200x?xf32>, %arg1: tensor<?x1200x16x1xf32>) -> tensor<?x1200x16x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %pack = linalg.pack %arg0
    padding_value(%cst : f32)
    outer_dims_perm = [1, 0]
    inner_dims_pos = [1, 0]
    inner_tiles = [16, 1]
    into %arg1 : tensor<1200x?xf32> -> tensor<?x1200x16x1xf32>
  return %pack : tensor<?x1200x16x1xf32>
}
// CHECK-LABEL: func @fold_padding_value_pack_negative2
// CHECK:         linalg.pack
// CHECK-SAME:      padding_value

// -----

func.func @fold_padding_value_pack_negative3(%arg0: tensor<1200x500000xf32>, %arg1: tensor<?x1200x?x1xf32>, %tile : index) -> tensor<?x1200x?x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %pack = linalg.pack %arg0
    padding_value(%cst : f32)
    outer_dims_perm = [1, 0]
    inner_dims_pos = [1, 0]
    inner_tiles = [%tile, 1]
    into %arg1 : tensor<1200x500000xf32> -> tensor<?x1200x?x1xf32>
  return %pack : tensor<?x1200x?x1xf32>
}
// CHECK-LABEL: func @fold_padding_value_pack_negative3
// CHECK:         linalg.pack
// CHECK-SAME:      padding_value

// -----

//===----------------------------------------------------------------------===//
// linalg.unpack
//===----------------------------------------------------------------------===//


// CHECK-LABEL: func @fold_unpack_constant_splat
//   CHECK-NOT: linalg.unpack
//       CHECK: arith.constant dense<1.000000e-01> : tensor<128x256xf32>
func.func @fold_unpack_constant_splat(%dest : tensor<128x256xf32>) -> tensor<128x256xf32> {
  %cst = arith.constant dense<1.000000e-01> : tensor<16x8x8x32xf32>
  %0 = linalg.unpack %cst inner_dims_pos = [0, 1]
    inner_tiles = [8, 32] into %dest : tensor<16x8x8x32xf32> -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

// -----

func.func @infer_dest_shape_unpack(%src: tensor<10x20x30x40x16xf32>, %dest: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %unpack = linalg.unpack %src
    outer_dims_perm = [2, 1, 3, 0]
    inner_dims_pos = [2]
    inner_tiles = [16]
    into %dest : tensor<10x20x30x40x16xf32> -> tensor<?x?x?x?xf32>
  return %unpack : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: func.func @infer_dest_shape_unpack
// CHECK-SAME:    %[[SRC:[0-9a-zA-Z]+]]
// CHECK-SAME:    %[[DEST:[0-9a-zA-Z]+]]
// CHECK:         %[[CAST_DEST:.+]] = tensor.cast %[[DEST]] : tensor<?x?x?x?xf32> to tensor<40x20x?x30xf32>
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[SRC]] {{.+}} into %[[CAST_DEST]]
// CHECK:         %[[CAST_UNPACK:.+]] = tensor.cast %[[UNPACK]] : tensor<40x20x?x30xf32> to tensor<?x?x?x?xf32>
// CHECK:         return %[[CAST_UNPACK]]

// -----

func.func @infer_src_shape_unpack(%src: tensor<?x?x?x?x16xf32>, %dest: tensor<30x20x?x10xf32>) -> tensor<30x20x?x10xf32> {
  %unpack = linalg.unpack %src
    outer_dims_perm = [2, 1, 3, 0]
    inner_dims_pos = [2]
    inner_tiles = [16]
    into %dest : tensor<?x?x?x?x16xf32> -> tensor<30x20x?x10xf32>
  return %unpack : tensor<30x20x?x10xf32>
}
// CHECK-LABEL: func.func @infer_src_shape_unpack
// CHECK-SAME:    %[[SRC:[0-9a-zA-Z]+]]
// CHECK-SAME:    %[[DEST:[0-9a-zA-Z]+]]
// CHECK:         %[[CAST_SRC:.+]] = tensor.cast %[[SRC]] : tensor<?x?x?x?x16xf32> to tensor<?x20x10x30x16xf32>
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[CAST_SRC]]
// CHECK:         return %[[UNPACK]]

// -----

func.func @no_infer_unpack_shape(%arg1: tensor<32x7x?x16x1xf32>, %arg2: index) -> tensor<?x32x100xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty(%arg2) : tensor<?x32x100xf32>
  %unpack = linalg.unpack %arg1 outer_dims_perm = [1, 2, 0] inner_dims_pos = [2, 0] inner_tiles = [16, 1] into %0 : tensor<32x7x?x16x1xf32> -> tensor<?x32x100xf32>
  return %unpack : tensor<?x32x100xf32>
}
// CHECK-LABEL: func.func @no_infer_unpack_shape
// CHECK-NOT:     tensor.cast

// -----

//===----------------------------------------------------------------------===//
// linalg.pack + linalg.unpack
//===----------------------------------------------------------------------===//

// Chain: NC -> NCnc -> NCnc -> NC
// CHECK: func.func @unpack_pack(
// CHECK-SAME: %[[T:.+]]: tensor<128x128xf32>)
// CHECK: return %[[T]] : tensor<128x128xf32>
func.func @unpack_pack(%t: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %tensor_empty = tensor.empty() : tensor<16x16x8x8xf32>
  %packed = linalg.pack %t inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %tensor_empty : tensor<128x128xf32> -> tensor<16x16x8x8xf32>
  %tensor_empty1 = tensor.empty() : tensor<128x128xf32>
  %unpacked = linalg.unpack %packed inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %tensor_empty1 : tensor<16x16x8x8xf32> -> tensor<128x128xf32>
  return %unpacked : tensor<128x128xf32>
}

// -----

// Chain: NC -> NCcn -> NCnc -> NC
// CHECK: func.func @unpack_pack(
// CHECK-SAME: %[[T:.+]]: tensor<128x128xf32>)
// CHECK-NOT: return %[[T]] : tensor<128x128xf32>
func.func @unpack_pack(%t: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %tensor_empty = tensor.empty() : tensor<16x16x8x8xf32>
  %packed = linalg.pack %t inner_dims_pos = [1, 0] inner_tiles = [8, 8] into %tensor_empty : tensor<128x128xf32> -> tensor<16x16x8x8xf32>
  %tensor_empty1 = tensor.empty() : tensor<128x128xf32>
  %unpacked = linalg.unpack %packed inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %tensor_empty1 : tensor<16x16x8x8xf32> -> tensor
<128x128xf32>
  return %unpacked : tensor<128x128xf32>
}

// -----

// Chain: NC -> CNcn -> NCnc -> NC
// CHECK: func.func @unpack_pack(
// CHECK-SAME: %[[T:.+]]: tensor<128x128xf32>)
// CHECK-NOT: return %[[T]] : tensor<128x128xf32>
func.func @unpack_pack(%t: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %tensor_empty = tensor.empty() : tensor<16x16x8x8xf32>
  %packed = linalg.pack %t outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 8] into %tensor_empty : tensor<128x128xf32> -> tensor<16x16x8x8xf32>
  %tensor_empty1 = tensor.empty() : tensor<128x128xf32>
  %unpacked = linalg.unpack %packed inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %tensor_empty1 : tensor<16x16x8x8xf32> -> tensor
<128x128xf32>
  return %unpacked : tensor<128x128xf32>
}

// -----

// Chain: NC -> NCnc -> NCnc -> NC
// CHECK: func.func @unpack_pack(
// CHECK-SAME: %[[T:.+]]: tensor<128x128xf32>,
// CHECK: return %[[T]] : tensor<128x128xf32>
func.func @unpack_pack(%t: tensor<128x128xf32>, %tile1: index, %tile2: index) -> tensor<128x128xf32> {
  %tensor_empty = tensor.empty(%tile1, %tile2) : tensor<16x16x?x?xf32>
  %packed = linalg.pack %t inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty : tensor<128x128xf32> -> tensor<16x16x?x?xf32>
  %tensor_empty1 = tensor.empty() : tensor<128x128xf32>
  %unpacked = linalg.unpack %packed inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty1 : tensor<16x16x?x?xf32> -> tensor
<128x128xf32>
  return %unpacked : tensor<128x128xf32>
}

// -----

// CHECK: func.func @unpack_pack_with_padding_no_canonicalization(
// CHECK:         linalg.pack
// CHECK:         linalg.unpack
func.func @unpack_pack_with_padding_no_canonicalization(%t: tensor<256x512xbf16>) -> tensor<224x512xbf16> {
  %tensor_empty = tensor.empty() : tensor<4x16x64x32xbf16>
  %tensor_empty1 = tensor.empty() : tensor<224x512xbf16>
  %packed = linalg.pack %t outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %tensor_empty : tensor<256x512xbf16> -> tensor<4x16x64x32xbf16>
  %unpacked = linalg.unpack %packed inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %tensor_empty1 : tensor<4x16x64x32xbf16> -> tensor<224x512xbf16>
  return %unpacked : tensor<224x512xbf16>
}

// -----

// Chain NCnc -> NC -> NC -> NCnc
// CHECK: func.func @pack_unpack(
// CHECK-SAME: %[[T:.+]]: tensor<16x16x?x?xf32>,
// CHECK: return %[[T]] : tensor<16x16x?x?xf32>
func.func @pack_unpack(%t: tensor<16x16x?x?xf32>, %tile1: index, %tile2: index) -> tensor<16x16x?x?xf32> {
  %tensor_empty = tensor.empty() : tensor<128x128xf32>
  %unpacked = linalg.unpack %t inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty : tensor<16x16x?x?xf32> -> tensor<128x128xf32>
  %tensor_empty1 = tensor.empty(%tile1, %tile2) : tensor<16x16x?x?xf32>
  %packed = linalg.pack %unpacked inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty1 : tensor<128x128xf32> -> tensor<16x16x?x?xf32>
  return %packed : tensor<16x16x?x?xf32>
}

// -----

// Chain NCnc -> NC -> NC -> NCnc
// CHECK: func.func @pack_unpack(
// CHECK-SAME: %[[T:.+]]: tensor<16x16x8x8xf32>
// CHECK: return %[[T]] : tensor<16x16x8x8xf32>
func.func @pack_unpack(%t: tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32> {
  %tensor_empty = tensor.empty() : tensor<128x128xf32>
  %unpacked = linalg.unpack %t inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %tensor_empty : tensor<16x16x8x8xf32> -> tensor<128x128xf32>
  %tensor_empty1 = tensor.empty() : tensor<16x16x8x8xf32>
  %packed = linalg.pack %unpacked inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %tensor_empty1 : tensor<128x128xf32> -> tensor<16x16x8x8xf32>
  return %packed : tensor<16x16x8x8xf32>
}

// -----

// CHECK: func.func @pack_unpack_same_tiles(
// CHECK-SAME:  %[[T:.+]]: tensor<?x?x?x?xf32>,
// CHECK: return %[[T]] : tensor<?x?x?x?xf32>
func.func @pack_unpack_same_tiles(%t: tensor<?x?x?x?xf32>, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index,
                       %tile1: index, %tile2: index) -> tensor<?x?x?x?xf32> {
  %tensor_empty = tensor.empty(%dim1, %dim2) : tensor<?x?xf32>
  %unpacked = linalg.unpack %t inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  %tensor_empty1 = tensor.empty(%dim3, %dim4, %dim5, %dim6) : tensor<?x?x?x?xf32>
  %packed = linalg.pack %unpacked inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty1 : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  return %packed : tensor<?x?x?x?xf32>
}

// -----

// CHECK: func.func @pack_unpack_different_tiles(
// CHECK-SAME:  %[[T:.+]]: tensor<?x?x?x?xf32>,
// CHECK-NOT: return %[[T]] : tensor<?x?x?x?xf32>
func.func @pack_unpack_different_tiles(%t: tensor<?x?x?x?xf32>, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index,
                       %tile1: index, %tile2: index) -> tensor<?x?x?x?xf32> {
  %tensor_empty = tensor.empty(%dim1, %dim2) : tensor<?x?xf32>
  %unpacked = linalg.unpack %t inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  %tensor_empty1 = tensor.empty(%dim3, %dim4, %dim5, %dim6) : tensor<?x?x?x?xf32>
  %packed = linalg.pack %unpacked inner_dims_pos = [0, 1] inner_tiles = [%tile2, %tile1] into %tensor_empty1 : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  return %packed : tensor<?x?x?x?xf32>
}

// -----

// CHECK: func.func @pack_unpack_dynamic_with_padding(
// CHECK-SAME:  %[[T:.+]]: tensor<?x?x?x?xf32>,
// CHECK-NOT: return %[[T]] : tensor<?x?x?x?xf32>
func.func @pack_unpack_dynamic_with_padding(%t: tensor<?x?x?x?xf32>, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index,
                       %tile1: index, %tile2: index, %pad: f32) -> tensor<?x?x?x?xf32> {
  %tensor_empty = tensor.empty(%dim1, %dim2) : tensor<?x?xf32>
  %unpacked = linalg.unpack %t inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  %tensor_empty1 = tensor.empty(%dim3, %dim4, %dim5, %dim6) : tensor<?x?x?x?xf32>
  %packed = linalg.pack %unpacked padding_value(%pad: f32) inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty1 : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  return %packed : tensor<?x?x?x?xf32>
}

// -----

// CHECK: func.func @pack_outer_dims_unpack_no_outer_dims(
// CHECK-SAME: %[[T:.+]]: tensor<16x16x?x?xf32>,
// CHECK: return %[[T]] : tensor<16x16x?x?xf32>
func.func @pack_outer_dims_unpack_no_outer_dims(%t: tensor<16x16x?x?xf32>, %tile1: index, %tile2: index) -> tensor<16x16x?x?xf32> {
  %tensor_empty = tensor.empty() : tensor<128x128xf32>
  %unpacked = linalg.unpack %t inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty : tensor<16x16x?x?xf32> -> tensor<128x128xf32>
  %tensor_empty1 = tensor.empty(%tile1, %tile2) : tensor<16x16x?x?xf32>
  %packed = linalg.pack %unpacked outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty1 : tensor<128x128xf32> -> tensor<16x16x?x?xf32>
  return %packed : tensor<16x16x?x?xf32>
}

// -----

// CHECK: func.func @pack_no_outer_dims_unpack_outer_dims(
// CHECK-SAME: %[[T:.+]]: tensor<16x16x?x?xf32>,
// CHECK: return %[[T]] : tensor<16x16x?x?xf32>
func.func @pack_no_outer_dims_unpack_outer_dims(%t: tensor<16x16x?x?xf32>, %tile1: index, %tile2: index) -> tensor<16x16x?x?xf32> {
  %tensor_empty = tensor.empty() : tensor<128x128xf32>
  %unpacked = linalg.unpack %t outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty : tensor<16x16x?x?xf32> -> tensor<128x128xf32>
  %tensor_empty1 = tensor.empty(%tile1, %tile2) : tensor<16x16x?x?xf32>
  %packed = linalg.pack %unpacked inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty1 : tensor<128x128xf32> -> tensor<16x16x?x?xf32>
  return %packed : tensor<16x16x?x?xf32>
}

// -----

//===----------------------------------------------------------------------===//
// tensor.cast + linalg.pack
//===----------------------------------------------------------------------===//

// CHECK-LABEL:   func.func @fold_cast_pack_dynamic_tile_size
// CHECK-SAME:      %[[DEST:.*]]: tensor<1x1x8x1xi32>,
// CHECK-SAME:      %[[SRC:.*]]: tensor<7x?xi32>,
// CHECK-SAME:      %[[PAD:.*]]: i32) -> tensor<1x1x8x1xi32> {
// CHECK:           %[[PACK:.*]] = linalg.pack %[[SRC]] padding_value(%[[PAD]] : i32)
// CHECK-SAME:        inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %[[DEST]]
// CHECK-SAME:        test_attr
// CHECK-SAME:        : tensor<7x?xi32> -> tensor<1x1x8x1xi32>
// CHECK:           return %[[PACK]] : tensor<1x1x8x1xi32>
func.func @fold_cast_pack_dynamic_tile_size(
  %dest: tensor<1x1x8x1xi32>,
  %src: tensor<7x?xi32>,
  %pad: i32) -> tensor<1x1x8x1xi32> {

    %cast = tensor.cast %dest : tensor<1x1x8x1xi32> to tensor<1x1x?x1xi32>
    %c8 = arith.constant 8 : index
    %pack = linalg.pack %src padding_value(%pad : i32)
      inner_dims_pos = [0, 1]
      inner_tiles = [%c8, 1]
      into %cast {test_attr} : tensor<7x?xi32> -> tensor<1x1x?x1xi32>
    %res = tensor.cast %pack : tensor<1x1x?x1xi32> to tensor<1x1x8x1xi32>
    return %res : tensor<1x1x8x1xi32>
}

// -----

func.func @infer_and_fold_pack_unpack_same_tiles(%t: tensor<10x20x4x4xf32>) -> tensor<10x20x4x4xf32> {
  %dim1 = arith.constant 40 : index
  %dim2 = arith.constant 80 : index
  %tensor_empty = tensor.empty(%dim1, %dim2) : tensor<?x?xf32>
  %unpacked = linalg.unpack %t inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %tensor_empty : tensor<10x20x4x4xf32> -> tensor<?x?xf32>
  %cast = tensor.cast %unpacked : tensor<?x?xf32> to tensor<40x80xf32>
  %tensor_empty1 = tensor.empty() : tensor<10x20x4x4xf32>
  %packed = linalg.pack %cast inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %tensor_empty1 : tensor<40x80xf32> -> tensor<10x20x4x4xf32>
  return %packed : tensor<10x20x4x4xf32>
}
// CHECK-LABEL: func.func @infer_and_fold_pack_unpack_same_tiles
// CHECK-SAME:    %[[SRC:[0-9a-zA-Z]+]]
// CHECK:         return %[[SRC]]

// -----

// CHECK-LABEL:   func.func @pack_dont_drop_attributes(
// CHECK: linalg.pack {{.*}}  {test_attr}
func.func @pack_dont_drop_attributes(%arg0: tensor<?x?x?xf16>, %arg1: tensor<128x?x100x16x1xf16>) -> tensor<128x?x100x16x1xf16> {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f16
  %pack = linalg.pack %arg0 padding_value(%cst : f16) outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [16, 1] into %arg1 {test_attr} : tensor<?x?x?xf16> -> tensor<128x?x100x16x1xf16>
  return %pack : tensor<128x?x100x16x1xf16>
}
// -----

//===----------------------------------------------------------------------===//
// linalg.fill + linalg.unpack
//===----------------------------------------------------------------------===//
// Fold DstStyleOp -> tensor.unpack operations.
func.func @fold_dst_style_ops_into_unpack(%arg0 : tensor<?x?x16x64xf32>, %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %fill : tensor<?x?x16x64xf32> -> tensor<?x?xf32>
  return %unpack : tensor<?x?xf32>
}
// CHECK-LABEL: func @fold_dst_style_ops_into_unpack
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x16x64xf32>
//  CHECK-SAME:     %[[INIT:.+]]: tensor<?x?xf32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[ARG0]]
//  CHECK-SAME:       into %[[INIT]]
//       CHECK:   return %[[UNPACK]]

// -----

//===----------------------------------------------------------------------===//
// tensor.cast + linalg.unpack
//===----------------------------------------------------------------------===//

// CHECK-LABEL:   func.func @fold_cast_unpack_dynamic_tile_size(
// CHECK-SAME:      %[[SRC:.*]]: tensor<1x1x8x1xi32>,
// CHECK-SAME:      %[[DEST:.*]]: tensor<7x?xi32>) -> tensor<7x?xi32> {
// CHECK:           %[[RES:.*]] = linalg.unpack %[[SRC]] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %[[DEST]] {test_attr} : tensor<1x1x8x1xi32> -> tensor<7x?xi32>
// CHECK:           return %[[RES]] : tensor<7x?xi32>
func.func @fold_cast_unpack_dynamic_tile_size(
  %src: tensor<1x1x8x1xi32>,
  %res: tensor<7x?xi32>) -> tensor<7x?xi32> {

    %cast = tensor.cast %src : tensor<1x1x8x1xi32> to tensor<1x1x?x1xi32>
    %c8 = arith.constant 8 : index
    %unpack = linalg.unpack %cast
      inner_dims_pos = [0, 1]
      inner_tiles = [%c8, 1]
      into %res {test_attr} : tensor<1x1x?x1xi32> -> tensor<7x?xi32>
    return %unpack : tensor<7x?xi32>
}
