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
//  CHECK-SAME:     %[[A:[a-z0-9]*]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[B:[a-z0-9]*]]: memref<?x?xf32>
//  CHECK-SAME:     %[[C:[a-z0-9]*]]: tensor<?x?xf32>
func.func @linalg_effects(%a : tensor<?x?xf32>, %b : memref<?x?xf32>, %c : tensor<?x?xf32>) {
  // CHECK-NOT:   %{{.*}} = linalg.matmul
  %t = linalg.matmul ins(%a, %b : tensor<?x?xf32>, memref<?x?xf32>)
                    outs(%c : tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK:   linalg.matmul
  linalg.matmul ins(%a, %c : tensor<?x?xf32>, tensor<?x?xf32>)
               outs(%b : memref<?x?xf32>)
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

func.func @propogate_casts(%arg0 : tensor<?x?xf32>, %arg1 : f32, %arg2 : index,
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
// CHECK-LABEL: func @propogate_casts
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
  %pack = tensor.pack %1 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %0 : tensor<384x512xf32> -> tensor<24x32x16x16xf32>
  return %pack : tensor<24x32x16x16xf32>
}
// CHECK-LABEL: func.func @fill_pack
// CHECK:         %[[PACKED_EMPTY:.+]] = tensor.empty() : tensor<24x32x16x16xf32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%{{.+}}) outs(%[[PACKED_EMPTY]]
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
  %pack = tensor.pack %0 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %3 : tensor<?x?xf32> -> tensor<?x?x16x16xf32>
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
func.func @identity_mixed(%arg0 : tensor<?xf32>, %arg1: memref<?xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<?xf32>)
    outs(%arg1 : memref<?xf32>) {
  ^bb0(%arg2 : f32, %arg3 : f32):
    linalg.yield %arg2 : f32
  }
  return
}

// There was a crash in EraseIdentityGenericOp for generic with mixed semantics.
// For now, check generic remained unchanged.
// CHECK-LABEL: func @identity_mixed
//  CHECK-SAME:     (%[[ARG1:.*]]: tensor<?xf32>, %[[ARG2:.*]]: memref<?xf32>)
//       CHECK:     linalg.generic {
//  CHECK-SAME:    indexing_maps = [#map, #map],
//  CHECK-SAME:    iterator_types = ["parallel"]
//  CHECK-SAME:  } ins(%[[ARG1]] : tensor<?xf32>)
//  CHECK-SAME:    outs(%[[ARG2]] : memref<?xf32>) {

// -----

// Just make sure that we don't crash.

// CHECK-LABEL: func @dedeplicate_regression_test
func.func @dedeplicate_regression_test(%0: tensor<4xf32>, %1: memref<4xf32>) {
  %36 = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0)>,
                      affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]}
    ins(%1, %1 : memref<4xf32>, memref<4xf32>)
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

#map = affine_map<(d0) -> (d0)>
func.func @cast_producer_mixed(%arg0 : tensor<5xf32>, %arg1: memref<?xf32>) {
  %0 = tensor.cast %arg0 : tensor<5xf32> to tensor<?xf32>
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]
  } ins(%0 : tensor<?xf32>)
    outs(%arg1 : memref<?xf32>) {
  ^bb0(%arg2 : f32, %arg3 : f32):
    linalg.yield %arg2 : f32
  }
  return
}

// We need a mixed linalg as a bridge between tensor and memref worlds.
// CHECK-LABEL: func @cast_producer_mixed
//  CHECK-SAME:     (%[[ARG1:.*]]: tensor<5xf32>, %[[ARG2:.*]]: memref<?xf32>)
//       CHECK:     linalg.generic {
//  CHECK-SAME:    indexing_maps = [#map, #map],
//  CHECK-SAME:    iterator_types = ["parallel"]
//  CHECK-SAME:  } ins(%[[ARG1]] : tensor<5xf32>)
//  CHECK-SAME:    outs(%[[ARG2]] : memref<?xf32>) {

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
