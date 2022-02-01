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

#map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @memref_cast_into_tiled_loop(
func @memref_cast_into_tiled_loop(%arg0: memref<192xf32>)  {
  %0 = memref.cast %arg0
    : memref<192xf32> to memref<192xf32, #map>
  %cst = arith.constant 0.000000e+00 : f32
  %c24 = arith.constant 24 : index
  %c0 = arith.constant 0 : index
  %c192 = arith.constant 192 : index
  // CHECK: linalg.tiled_loop
  // CHECK-SAME: outs (%{{.*}} = %{{.*}}: memref<192xf32>)
  linalg.tiled_loop (%arg3) = (%c0) to (%c192) step (%c24)
    outs (%out = %0: memref<192xf32, #map>) {
    %14 = affine.min affine_map<(d0) -> (-d0 + 192, 24)>(%arg3)
    %16 = memref.subview %out[%arg3] [%14] [1]
      : memref<192xf32, #map> to memref<?xf32, #map>
    linalg.fill(%cst, %16) : f32, memref<?xf32, #map>
    linalg.yield
  }
  return
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
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
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

//   CHECK-NOT: linalg.copy
  linalg.copy(%arg0, %arg0): memref<2x3x?x4xf32>, memref<2x3x?x4xf32>

//   CHECK: return
  return
}

// -----

// CHECK-LABEL: @self_copy_with_permutation
func @self_copy_with_permutation(%arg0 : memref<2x3x?x4xf32>) {

//   CHECK: linalg.copy
  linalg.copy(%arg0, %arg0)
    {inputPermutation = affine_map<(i, j, k, l) -> (j, k, i, l)>,
     outputPermuation = affine_map<(i, j, k, l) -> (i, j, k, l)>} : memref<2x3x?x4xf32>, memref<2x3x?x4xf32>

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

func private @foo(%A: memref<48xf32>, %B: tensor<48xf32>,
                  %C: memref<48xf32>) -> (tensor<48xf32>)

func @fold_tiled_loop_results(%A: memref<48xf32>, %B: tensor<48xf32>,
    %C: memref<48xf32>, %C_tensor: tensor<48xf32>) -> tensor<48xf32> {
  %c0 = arith.constant 0 : index
  %c24 = arith.constant 24 : index
  %c48 = arith.constant 48 : index
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

// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C24:.*]] = arith.constant 24 : index
// CHECK-DAG:  %[[C48:.*]] = arith.constant 48 : index

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
  %c0 = arith.constant 0 : index
  %c24 = arith.constant 24 : index
  %c192 = arith.constant 192 : index
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

// CHECK-LABEL: func @dim_of_tiled_loop_input_no_canonicalize(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[arg2:.*]]: tensor<?x?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   linalg.tiled_loop {{.*}} outs (%[[o:.*]] =
//       CHECK:     %[[dim:.*]] = tensor.dim %[[o]], %[[c0]]
//       CHECK:     arith.index_cast %[[dim]]
func @dim_of_tiled_loop_input_no_canonicalize(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %s: index)
    -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %r = linalg.tiled_loop (%iv0, %iv1) = (%c0, %c0)
      to (%d0, %d1) step (%c1, %c1)
      ins (%in0 = %arg0 : tensor<?x?xf32>, %in1 = %arg1 : tensor<?x?xf32>)
      outs (%out1 = %arg2 : tensor<?x?xf32>) {
    %inner_dim = tensor.dim %out1, %c0 : tensor<?x?xf32>
    %cast1 = arith.index_cast %inner_dim : index to i32
    %cast2 = arith.sitofp %cast1 : i32 to f32
    %fill = linalg.fill(%cast2, %out1) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
    %slice = tensor.extract_slice %fill[0, 0][%s, %s][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    linalg.yield %slice : tensor<?x?xf32>
  }
  return %r : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @dim_of_tiled_loop_input(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[arg2:.*]]: tensor<?x?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   linalg.tiled_loop
//       CHECK:     %[[dim:.*]] = tensor.dim %[[arg1]], %[[c0]]
//       CHECK:     arith.index_cast %[[dim]]
func @dim_of_tiled_loop_input(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %r = linalg.tiled_loop (%iv0, %iv1) = (%c0, %c0)
      to (%d0, %d1) step (%c1, %c1)
      ins (%in0 = %arg0 : tensor<?x?xf32>, %in1 = %arg1 : tensor<?x?xf32>)
      outs (%out1 = %arg2 : tensor<?x?xf32>) {
    %inner_dim = tensor.dim %in1, %c0 : tensor<?x?xf32>
    %cast1 = arith.index_cast %inner_dim : index to i32
    %cast2 = arith.sitofp %cast1 : i32 to f32
    %fill = linalg.fill(%cast2, %out1) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
    linalg.yield %fill : tensor<?x?xf32>
  }
  return %r : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @dim_of_tiled_loop_result(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[arg2:.*]]: tensor<?x?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   tensor.dim %[[arg2]], %[[c0]]
func @dim_of_tiled_loop_result(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %s: index)
    -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %r = linalg.tiled_loop (%iv0, %iv1) = (%c0, %c0)
      to (%d0, %d1) step (%c1, %c1)
      ins (%in0 = %arg0 : tensor<?x?xf32>, %in1 = %arg1 : tensor<?x?xf32>)
      outs (%out1 = %arg2 : tensor<?x?xf32>) {
    %1 = tensor.insert_slice %arg0 into %out1 [0, 0] [%s, %s] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
    linalg.yield %1 : tensor<?x?xf32>
  }
  %r2 = tensor.dim %r, %c0 : tensor<?x?xf32>
  return %r2 : index
}

// -----

// CHECK-LABEL: func @dim_of_tiled_loop_result_no_canonicalize(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[arg2:.*]]: tensor<?x?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[r:.*]] = linalg.tiled_loop
//       CHECK:   tensor.dim %[[r]], %[[c0]]
func @dim_of_tiled_loop_result_no_canonicalize(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %s: index)
    -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %r = linalg.tiled_loop (%iv0, %iv1) = (%c0, %c0)
      to (%d0, %d1) step (%c1, %c1)
      ins (%in0 = %arg0 : tensor<?x?xf32>, %in1 = %arg1 : tensor<?x?xf32>)
      outs (%out1 = %arg2 : tensor<?x?xf32>) {
    %1 = tensor.insert_slice %arg0 into %arg1 [0, 0] [%s, %s] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
    linalg.yield %1 : tensor<?x?xf32>
  }
  %r2 = tensor.dim %r, %c0 : tensor<?x?xf32>
  return %r2 : index
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
