// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func @memref_cast(
func @memref_cast(%a: index, %b: index) -> memref<?x?xf32> {
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

func @dce_zero_memref(%arg0 : memref<0xf32>, %arg1: tensor<0xf32>) -> tensor<0xf32> {
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
  %c6 = arith.constant 6 : index
  %0 = linalg.init_tensor [4, 5, %c6] : tensor<4x5x?xf32>
  return %0 : tensor<4x5x?xf32>
}
// CHECK: func @init_tensor_canonicalize
// CHECK:   %[[T0:.+]] = linalg.init_tensor [4, 5, 6] : tensor<4x5x6xf32>
// CHECK:   %[[T1:.+]] = tensor.cast %[[T0]] : tensor<4x5x6xf32> to tensor<4x5x?xf32>
// CHECK:   return %[[T1]]

// -----

func @init_tensor_reshape_expansion(%arg0 : index) -> tensor<2x3x5x4x?x7xf32> {
  %0 = linalg.init_tensor [6, 5, %arg0] : tensor<6x5x?xf32>
  %1 = tensor.expand_shape %0 [[0, 1], [2], [3, 4, 5]]
      : tensor<6x5x?xf32> into tensor<2x3x5x4x?x7xf32>
  return %1 : tensor<2x3x5x4x?x7xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 28)>
//      CHECK: func @init_tensor_reshape_expansion
// CHECK-SAME:     %[[ARG0:.+]]: index
// CHECK-NEXT:   %[[D:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
// CHECK-NEXT:   %[[INIT:.+]] = linalg.init_tensor [2, 3, 5, 4, %[[D]], 7]
// CHECK-NEXT:   return %[[INIT]]

// -----

func @init_tensor_reshape_collapse(%arg0 : index) -> tensor<6x5x?xf32> {
  %0 = linalg.init_tensor [2, 3, 5, 4, %arg0, 7] : tensor<2x3x5x4x?x7xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2], [3, 4, 5]]
      : tensor<2x3x5x4x?x7xf32> into tensor<6x5x?xf32>
  return %1 : tensor<6x5x?xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 28)>
//      CHECK: func @init_tensor_reshape_collapse
// CHECK-SAME:     %[[ARG0:.+]]: index
// CHECK-NEXT:   %[[D:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
// CHECK-NEXT:   %[[INIT:.+]] = linalg.init_tensor [6, 5, %[[D]]]
// CHECK-NEXT:   return %[[INIT]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @remove_no_op(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>)
  -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = linalg.init_tensor [%0, %1, %2] : tensor<?x?x?xf32>
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
func @remove_no_op_mismatched_types(%arg0 : tensor<?x?x?xf32>)
  -> tensor<1x2x3xf32> {
  %out = linalg.init_tensor [1, 2, 3] : tensor<1x2x3xf32>
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

#map = affine_map<(d0, d1) -> (d0, d1)>
func @keep_not_noop(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
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
func @keep_not_noop(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
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

func @fold_init_tensor_with_slice
  (%arg0 : index, %arg1 : index) -> tensor<5x?x20xf32>
{
  %0 = linalg.init_tensor[%arg0, 10, 40] : tensor<?x10x40xf32>
  %1 = tensor.extract_slice %0[0, 0, 0] [5, %arg1, 20] [1, 1, 1]
    : tensor<?x10x40xf32> to tensor<5x?x20xf32>
  return %1 : tensor<5x?x20xf32>
}
//      CHECK: func @fold_init_tensor_with_slice
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//      CHECK:   %[[T0:.+]] = linalg.init_tensor [5, %[[ARG1]], 20]
//      CHECK:   return %[[T0]]

// -----

func @fold_init_tensor_with_cast(%arg0 : index) -> tensor<1x12xf32> {
  %0 = linalg.init_tensor [%arg0, 12] : tensor<?x12xf32>
  %1 = tensor.cast %0 : tensor<?x12xf32> to tensor<1x12xf32>
  return %1 : tensor<1x12xf32>
}
//      CHECK: func @fold_init_tensor_with_cast(%[[ARG0:.+]]: index)
//      CHECK:   %[[T0:.+]] = linalg.init_tensor [1, 12] : tensor<1x12xf32>
//      CHECK:   return %[[T0]] : tensor<1x12xf32>

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
func @dead_linalg_tensor(%arg0 : tensor<7x7xi32>, %arg1 : tensor<7x7xf32>,
                         %arg2: tensor<?x?xf32>, %high : index) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill(%c0_i32, %arg0) : i32, tensor<7x7xi32> -> tensor<7x7xi32>
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

func @propogate_casts(%arg0 : tensor<?x?xf32>, %arg1 : f32, %arg2 : index,
    %arg3 : index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c21 = arith.constant 21 : index
  %c42 = arith.constant 42 : index
  %0 = linalg.init_tensor [%c21, %c42] : tensor<?x?xf32>
  %1 = linalg.fill(%arg1, %0) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
  %2 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %3 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %4 = tensor.insert_slice %arg0 into %1[%arg2, %arg3] [%2, %3] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
// CHECK-LABEL: func @propogate_casts
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor [21, 42]
//       CHECK:   %[[FILL:.+]] = linalg.fill(%{{.+}}, %[[INIT]])
//       CHECK:   %[[INSERTED:.+]] = tensor.insert_slice %{{.+}} into %[[FILL]]
//       CHECK:   %[[RESULT:.+]] = tensor.cast %[[INSERTED]]
//       CHECK:   return %[[RESULT]]

// -----

// CHECK-LABEL: @self_copy
func @self_copy(%arg0 : memref<2x3x?x4xf32>) {

//   CHECK-NOT: memref.copy
  memref.copy %arg0, %arg0 : memref<2x3x?x4xf32> to memref<2x3x?x4xf32>

//   CHECK: return
  return
}

// -----

// CHECK-LABEL: func @fold_fill_reshape()
func @fold_fill_reshape() -> tensor<6x4xf32> {
  %zero = arith.constant 0.0 : f32
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [6, 4] : tensor<6x4xf32>
  %init = linalg.init_tensor [1, 2, 3, 4] : tensor<1x2x3x4xf32>
  // CHECK: %[[FILL:.+]] = linalg.fill(%cst, %[[INIT]]) : f32, tensor<6x4xf32> -> tensor<6x4xf32>
  %fill = linalg.fill(%zero, %init) : f32, tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32>
  %reshape = tensor.collapse_shape %fill [[0, 1, 2], [3]]
      : tensor<1x2x3x4xf32> into tensor<6x4xf32>
  // CHECK: return %[[FILL]] : tensor<6x4xf32>
  return %reshape : tensor<6x4xf32>
}

// -----

//       CHECK: func @fold_fill_reshape_dynamic
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?x?x?xf32>
func @fold_fill_reshape_dynamic(%arg0 : tensor<?x?x?x?x?xf32>) -> tensor<?x?xf32> {
  %zero = arith.constant 0.0 : f32
  // CHECK: %[[RESHAPE:.+]] = tensor.collapse_shape %[[ARG0]]
  %0 = linalg.fill(%zero, %arg0) : f32, tensor<?x?x?x?x?xf32> -> tensor<?x?x?x?x?xf32>
  // CHECK: %[[RESULT:.+]] = linalg.fill(%{{.+}}, %[[RESHAPE]])
  %1 = tensor.collapse_shape %0 [[0, 1, 2], [3, 4]]
      : tensor<?x?x?x?x?xf32> into tensor<?x?xf32>
  // CHECK: return %[[RESULT]]
  return %1 : tensor<?x?xf32>
}


// -----

func private @some_use(%i : index, %j : index)

// CHECK-LABEL: func @init_canonicalize
//  CHECK-SAME:   %[[I:.*]]: index
func @init_canonicalize(%i : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK-NOT: init_tensor
  %0 = linalg.init_tensor [%i, 42] : tensor<?x42xf32>

  // CHECK-NOT: tensor.dim
  %1 = tensor.dim %0, %c0: tensor<?x42xf32>
  %2 = tensor.dim %0, %c1: tensor<?x42xf32>

  // CHECK: %[[c42:.*]] = arith.constant 42 : index
  // CHECK: call @some_use(%[[I]], %[[c42]])
  call @some_use(%1, %2) : (index, index) -> ()

  return
}

// -----

// CHECK-LABEL: func @rank_reducing_init_extract
func @rank_reducing_init_extract(%sz : index, %idx : index) -> tensor<2xf32> {
  // CHECK: linalg.init_tensor [2] : tensor<2xf32>
  %a = linalg.init_tensor [%sz, 2] : tensor<?x2xf32>

  // CHECK-NOT: extract
  %r = tensor.extract_slice %a[%idx, 0] [1, 2] [1, 1] : tensor<?x2xf32> to tensor<2xf32>
  return %r: tensor<2xf32>
}

// -----

// CHECK: func @fold_self_copy
func @fold_self_copy(%0 : memref<4x16xf32>) {
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
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor [412, 276] : tensor<412x276xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill(%[[F0]], %[[INIT]])
//       CHECK:   return %[[FILL]]
func @fold_static_pad_fill() -> tensor<412x276xf32> {
  %f0 = arith.constant 0.0 : f32
  %init = linalg.init_tensor [400, 273] : tensor<400x273xf32>
  %fill = linalg.fill(%f0, %init) : f32, tensor<400x273xf32> -> tensor<400x273xf32>
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
//      CHECK:   %[[OF:.+]] = linalg.fill(%[[F0]], %[[SRC]]) : f32, tensor<8x?x16x32xf32>
//      CHECK:   %[[S0:.+]] = affine.apply #[[MAP0]]()[%[[LOW0]]]
//      CHECK:   %[[DIM1:.+]] = tensor.dim %[[OF]], %[[I1]] : tensor<8x?x16x32xf32>
//      CHECK:   %[[S1:.+]] = affine.apply #[[MAP1]]()[%[[DIM1]]]
//      CHECK:   %[[S2:.+]] = affine.apply #[[MAP2]]()[%[[HIGH2]]]
//      CHECK:   %[[S3:.+]] = affine.apply #[[MAP3]]()[%[[LOW3]], %[[HIGH3]]]
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[S0]], %[[S1]], %[[S2]], %[[S3]]] : tensor<?x?x?x?xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill(%[[F0]], %[[INIT]])
//      CHECK:   return %[[FILL]]
func @fold_dynamic_pad_fill(%init: tensor<8x?x16x32xf32>, %low0: index, %low3: index, %high2: index, %high3: index) -> tensor<?x?x?x?xf32> {
  %f0 = arith.constant 0.0 : f32
  %fill = linalg.fill(%f0, %init) : f32, tensor<8x?x16x32xf32> -> tensor<8x?x16x32xf32>
  %pad = tensor.pad %fill low[%low0, 8, 7, %low3] high[1, 2, %high2, %high3] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %f0 : f32
  } : tensor<8x?x16x32xf32> to tensor<?x?x?x?xf32>
  return %pad : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @no_fold_pad_fill_value_mismatch
func @no_fold_pad_fill_value_mismatch() -> tensor<412x276xf32> {
  %f0 = arith.constant 0.0 : f32
  %f1 = arith.constant 1.0 : f32
  %init = linalg.init_tensor [400, 273] : tensor<400x273xf32>
  %fill = linalg.fill(%f0, %init) : f32, tensor<400x273xf32> -> tensor<400x273xf32>
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
func @static_input_without_cast(%arg0 : tensor<2x3x4xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<2x3x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<2x3x4xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<2x3x4xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<2x3x4xf32>
  %3 = linalg.init_tensor [%0, %1, %2] : tensor<?x?x?xf32>
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
func @static_input_with_cast(%arg0 : tensor<2x3x4xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<2x3x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<2x3x4xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<2x3x4xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<2x3x4xf32>
  %3 = linalg.init_tensor [%0, %1, %2] : tensor<?x?x?xf32>
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
func @static_output_with_cast(%arg0 : tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg2, %c0 : tensor<2x3x4xf32>
  %1 = tensor.dim %arg2, %c1 : tensor<2x3x4xf32>
  %2 = tensor.dim %arg2, %c2 : tensor<2x3x4xf32>
  %3 = linalg.init_tensor [%0, %1, %2] : tensor<?x?x?xf32>
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
func @cast_source(%arg0 : tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<2x3x4xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<2x3x4xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<2x3x4xf32>
  %3 = linalg.init_tensor [%0, %1, %2] : tensor<?x?x?xf32>
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
func @cast_dest(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x?x?xf32>, %arg2: index, %arg3: index, %arg4: index) -> tensor<?x?x?xf32> {
  %0 = linalg.init_tensor [%arg2, %arg3, %arg4] : tensor<?x?x?xf32>
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
//       CHECK: %[[INIT:.+]] = linalg.init_tensor [8, 384, 384]
//       CHECK: %[[FILL:.+]] = linalg.fill(%[[F0]], %[[INIT]])
//       CHECK: %[[OFFSET1:.+]] = affine.apply #[[$MAP]]()[%[[LOW1]]]
//       CHECK: %[[D0:.+]] = tensor.dim %[[INPUT]], %[[C0]] : tensor<?x?x?xf32>
//       CHECK: %[[D1:.+]] = tensor.dim %[[INPUT]], %[[C1]] : tensor<?x?x?xf32>
//       CHECK: %[[D2:.+]] = tensor.dim %[[INPUT]], %[[C2]] : tensor<?x?x?xf32>
//       CHECK: tensor.insert_slice %[[INPUT]] into %[[FILL]][%[[LOW0]], %[[OFFSET1]], 2] [%[[D0]], %[[D1]], %[[D2]]] [1, 1, 1]
func @insert_pad_into_fill(%input: tensor<?x?x?xf32>, %low0: index, %low1: index, %high1: index, %high2: index) -> tensor<8x384x384xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %pad = tensor.pad %input low[%low0, %low1, %c0] high[%c0, %high1, %high2] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %f0 : f32
  } : tensor<?x?x?xf32> to tensor<8x128x128xf32>
  %init = linalg.init_tensor [8, 384, 384] : tensor<8x384x384xf32>
  %fill = linalg.fill(%f0, %init) : f32, tensor<8x384x384xf32> -> tensor<8x384x384xf32>
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
func @multi_insert_pad_into_fill(%input: tensor<7x123x124xf32>, %a: tensor<8x128x128xf32>, %offset: index) -> tensor<8x384x384xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %pad = tensor.pad %input low[1, 2, 0] high[0, 3, 4] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %f0 : f32
  } : tensor<7x123x124xf32> to tensor<8x128x128xf32>
  %init = linalg.init_tensor [8, 384, 384] : tensor<8x384x384xf32>
  %fill = linalg.fill(%f0, %init) : f32, tensor<8x384x384xf32> -> tensor<8x384x384xf32>
  %0 = tensor.insert_slice %a   into %fill[%offset, 0, 0]  [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %1 = tensor.insert_slice %a   into %0   [0, 128, %offset][8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %2 = tensor.insert_slice %pad into %1   [0, 0, 256]      [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  return %2: tensor<8x384x384xf32>
}

// -----

// CHECK-LABEL: func @multi_insert_pad_into_fill_overlap
func @multi_insert_pad_into_fill_overlap(%input: tensor<7x123x124xf32>, %a: tensor<8x128x128xf32>, %offset: index) -> tensor<8x384x384xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK: tensor.pad
  %pad = tensor.pad %input low[1, 2, 0] high[0, 3, 4] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %f0 : f32
  } : tensor<7x123x124xf32> to tensor<8x128x128xf32>
  %init = linalg.init_tensor [8, 384, 384] : tensor<8x384x384xf32>
  %fill = linalg.fill(%f0, %init) : f32, tensor<8x384x384xf32> -> tensor<8x384x384xf32>
  %0 = tensor.insert_slice %a   into %fill[%offset, 0, 0]  [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %1 = tensor.insert_slice %a   into %0   [0, 0, 129]      [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  // Range overlap with %1 at dim#3
  %2 = tensor.insert_slice %pad into %1   [0, 0, 256]      [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  return %2: tensor<8x384x384xf32>
}

// -----

// CHECK-LABEL: func @multi_insert_pad_into_fill_overlap
func @multi_insert_pad_into_fill_overlap(%input: tensor<7x123x124xf32>, %a: tensor<8x128x128xf32>, %offset: index) -> tensor<8x384x384xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK: tensor.pad
  %pad = tensor.pad %input low[1, 2, 0] high[0, 3, 4] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %f0 : f32
  } : tensor<7x123x124xf32> to tensor<8x128x128xf32>
  %init = linalg.init_tensor [8, 384, 384] : tensor<8x384x384xf32>
  %fill = linalg.fill(%f0, %init) : f32, tensor<8x384x384xf32> -> tensor<8x384x384xf32>
  %0 = tensor.insert_slice %a   into %fill[0, 0, %offset]  [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %1 = tensor.insert_slice %a   into %0   [0, 128, 255]    [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  // Range overlap with %0 at dim#3
  %2 = tensor.insert_slice %pad into %1   [0, 0, 256]      [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  return %2: tensor<8x384x384xf32>
}

// -----

// CHECK-LABEL: func @multi_insert_pad_into_fill
func @multi_insert_pad_into_fill(%input: tensor<7x123x124xf32>, %a: tensor<8x128x128xf32>, %offset: index) -> tensor<8x384x384xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK-NOT: tensor.pad
  %pad = tensor.pad %input low[1, 2, 0] high[0, 3, 4] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %f0 : f32
  } : tensor<7x123x124xf32> to tensor<8x128x128xf32>
  %init = linalg.init_tensor [8, 384, 384] : tensor<8x384x384xf32>
  %fill = linalg.fill(%f0, %init) : f32, tensor<8x384x384xf32> -> tensor<8x384x384xf32>
  // Overlap btween %0 and %1 is fine but not with %2 is fine.
  // CHECK-COUNT-3: tensor.insert_slice
  %0 = tensor.insert_slice %a   into %fill[0, 0, %offset]  [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %1 = tensor.insert_slice %a   into %0   [0, 1, %offset]  [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %2 = tensor.insert_slice %pad into %1   [0, 256, 256]    [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  return %2: tensor<8x384x384xf32>
}

// -----

// CHECK-LABEL: func @multi_insert_pad_into_fill_mismatch
func @multi_insert_pad_into_fill_mismatch(%input: tensor<7x123x124xf32>, %a: tensor<8x128x128xf32>, %offset: index) -> tensor<8x384x384xf32> {
  %f0 = arith.constant 0.0 : f32
  %f1 = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK: tensor.pad
  %pad = tensor.pad %input low[1, 2, 0] high[0, 3, 4] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %f0 : f32
  } : tensor<7x123x124xf32> to tensor<8x128x128xf32>
  %init = linalg.init_tensor [8, 384, 384] : tensor<8x384x384xf32>
  // Different filling value than padding value.
  %fill = linalg.fill(%f1, %init) : f32, tensor<8x384x384xf32> -> tensor<8x384x384xf32>
  %0 = tensor.insert_slice %a   into %fill[%offset, 0, 0]  [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %1 = tensor.insert_slice %a   into %0   [0, 128, %offset][8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  %2 = tensor.insert_slice %pad into %1   [0, 0, 256]      [8, 128, 128] [1, 1, 1] : tensor<8x128x128xf32> into tensor<8x384x384xf32>
  return %2: tensor<8x384x384xf32>
}

// -----

func @fold_linalgop_with_cast_consumer(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
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

func @fold_conv_op_with_cast_consumer(%arg0 : tensor<?x?x?x?xf32>,
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
