// RUN: mlir-opt  -transform-interpreter -split-input-file --cse %s | FileCheck %s

func.func @simple_matmul(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.tile_using_forall %matmul [10, 20] mapping = [#gpu.block<y>, #gpu.block<x>]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (10, -d0 + s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (20, -d0 + s0)>
//      CHECK: func.func @simple_matmul(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[RESULT:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) =
// CHECK-SAME:       (0, 0) to (%[[M]], %[[N]]) step (10, 20) shared_outs(%[[INIT:.+]] = %[[ARG2]])
//      CHECK:     %[[TS_Y:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M]]]
//      CHECK:     %[[TS_X:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[N]]]
//      CHECK:     %[[LHS_TILE:.+]] = tensor.extract_slice %[[ARG0]]
// CHECK-SAME:         [%[[IV0]], 0] [%[[TS_Y]], %[[K]]] [1, 1]
//      CHECK:     %[[RHS_TILE:.+]] = tensor.extract_slice %[[ARG1]]
// CHECK-SAME:         [0, %[[IV1]]] [%[[K]], %[[TS_X]]] [1, 1]
//      CHECK:     %[[INIT_TILE:.+]] = tensor.extract_slice %[[INIT]]
// CHECK-SAME:         [%[[IV0]], %[[IV1]]] [%[[TS_Y]], %[[TS_X]]] [1, 1]
//      CHECK:     %[[GEMM_TILE:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS_TILE]], %[[RHS_TILE]] :
// CHECK-SAME:         outs(%[[INIT_TILE]] :
//      CHECK:     scf.forall.in_parallel {
//      CHECK:       tensor.parallel_insert_slice %[[GEMM_TILE]] into %[[INIT]]
// CHECK-SAME:           [%[[IV0]], %[[IV1]]] [%[[TS_Y]], %[[TS_X]]] [1, 1]
//      CHECK:       mapping = [#gpu.block<y>, #gpu.block<x>]
//      CHECK:   return %[[RESULT]]

// -----

func.func @simple_matmul_memref(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>,
    %arg2 : memref<?x?xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%arg2 : memref<?x?xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.tile_using_forall %matmul [10, 20]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (10, -d0 + s0)>
//  CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (20, -d0 + s0)>
//      CHECK-LABEL: func.func @simple_matmul_memref(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: memref<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: memref<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[M:.+]] = memref.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = memref.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[N:.+]] = memref.dim %[[ARG1]], %[[C1]]
//      CHECK:   scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) = (0, 0) to (%[[M]], %[[N]]) step (10, 20) {
//  CHECK-DAG:     %[[TS_M:.+]] = affine.min #[[$MAP0]](%[[IV0]])[%[[M]]]
//  CHECK-DAG:     %[[TS_N:.+]] = affine.min #[[$MAP1]](%[[IV1]])[%[[N]]]
//  CHECK-DAG:     %[[LHS_TILE:.+]] = memref.subview %[[ARG0]]
// CHECK-SAME:         [%[[IV0]], 0] [%[[TS_M]], %[[K]]] [1, 1]
//  CHECK-DAG:     %[[RHS_TILE:.+]] = memref.subview %[[ARG1]]
// CHECK-SAME:         [0, %[[IV1]]] [%[[K]], %[[TS_N]]] [1, 1]
//  CHECK-DAG:     %[[OUT_TILE:.+]] = memref.subview %[[ARG2]]
// CHECK-SAME:         [%[[IV0]], %[[IV1]]] [%[[TS_M]], %[[TS_N]]] [1, 1]
//      CHECK:     linalg.matmul
// CHECK-SAME:             ins(%[[LHS_TILE]], %[[RHS_TILE]] :
// CHECK-SAME:             outs(%[[OUT_TILE]] :

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
func.func @multi_result(%arg0 : tensor<128x200x300xf32>) -> (tensor<128x300x200xf32>, tensor<300x128x200xf32>) {
  %init0 = tensor.empty() : tensor<128x300x200xf32>
  %init1 = tensor.empty() : tensor<300x128x200xf32>
  %0:2 = linalg.generic {
      indexing_maps = [#map0, #map1, #map2],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<128x200x300xf32>)
      outs(%init0, %init1 : tensor<128x300x200xf32>, tensor<300x128x200xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      linalg.yield %b0, %b0 : f32, f32
    } -> (tensor<128x300x200xf32>, tensor<300x128x200xf32>)
  return %0#0, %0#1 : tensor<128x300x200xf32>, tensor<300x128x200xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.tile_using_forall %generic [10, 0, 20]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0) -> (10, -d0 + 128)>
//      CHECK-LABEL: func.func @multi_result(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<128x200x300xf32>)
//  CHECK-DAG:   %[[INIT0:.+]] = tensor.empty()
//  CHECK-DAG:   %[[INIT1:.+]] = tensor.empty()
//      CHECK:   %[[OUTER:[a-zA-Z0-9]+]]:2 = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) = (0, 0) to (128, 300) step (10, 20)
// CHECK-SAME:       shared_outs(%[[ARG1:[a-zA-Z0-9]+]] = %[[INIT0]], %[[ARG2:[a-zA-Z0-9]+]] = %[[INIT1]])
//      CHECK:     %[[TS_Y:.+]] = affine.min #[[$MAP0]](%[[IV0]])
//      CHECK:     %[[ARG_TILE:.+]] = tensor.extract_slice %[[ARG0]]
// CHECK-SAME:         [%[[IV0]], 0, %[[IV1]]] [%[[TS_Y]], 200, 20] [1, 1, 1]
//  CHECK-DAG:     %[[INIT0_TILE:.+]] = tensor.extract_slice %[[ARG1]]
// CHECK-SAME:         [%[[IV0]], %[[IV1]], 0] [%[[TS_Y]], 20, 200] [1, 1, 1]
//  CHECK-DAG:     %[[INIT1_TILE:.+]] = tensor.extract_slice %[[ARG2]]
// CHECK-SAME:         [%[[IV1]], %[[IV0]], 0] [20, %[[TS_Y]], 200] [1, 1, 1]
//      CHECK:     %[[RESULT_TILE:.+]]:2 = linalg.generic
// CHECK-SAME:         ins(%[[ARG_TILE]] :
// CHECK-SAME:         outs(%[[INIT0_TILE]], %[[INIT1_TILE]] :
//      CHECK:     scf.forall.in_parallel {
//  CHECK-DAG:       tensor.parallel_insert_slice %[[RESULT_TILE]]#0 into %[[ARG1]][%[[IV0]], %[[IV1]], 0] [%[[TS_Y]], 20, 200] [1, 1, 1]
//  CHECK-DAG:       tensor.parallel_insert_slice %[[RESULT_TILE]]#1 into %[[ARG2]][%[[IV1]], %[[IV0]], 0] [20, %[[TS_Y]], 200] [1, 1, 1]
//      CHECK:     }
//      CHECK:   return %[[OUTER]]#0, %[[OUTER]]#1

// -----

func.func @conv2D(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x?x?xf32>,
    %arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf {
      strides = dense<[2, 3]> : tensor<2xi64>,
      dilation = dense<[4, 5]> : tensor<2xi64>}
      ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %conv = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.tile_using_forall %conv [0, 0, 0, 0, 10, 20, 30]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (10, -d0 + s0)>
//  CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (20, -d0 + s0)>
//  CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0)[s0] -> (30, -d0 + s0)>
//  CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0)[s0] -> (d0 + s0 * 2 - 2)>
//  CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0)[s0] -> (d0 + s0 * 3 - 3)>
//      CHECK-LABEL: func.func @conv2D(
// CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:     %[[FILTER:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[INPUT]], %[[C0]]
//  CHECK-DAG:   %[[C:.+]] = tensor.dim %[[INPUT]], %[[C3]]
//  CHECK-DAG:   %[[P:.+]] = tensor.dim %[[FILTER]], %[[C0]]
//  CHECK-DAG:   %[[Q:.+]] = tensor.dim %[[FILTER]], %[[C1]]
//  CHECK-DAG:   %[[F:.+]] = tensor.dim %[[FILTER]], %[[C3]]
//  CHECK-DAG:   %[[R:.+]] = tensor.dim %[[INIT]], %[[C1]]
//  CHECK-DAG:   %[[S:.+]] = tensor.dim %[[INIT]], %[[C2]]
//      CHECK:   %[[RESULT:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]], %[[IV2:[a-zA-Z0-9]+]]) =
// CHECK-SAME:       (0, 0, 0) to (%[[P]], %[[Q]], %[[C]]) step (10, 20, 30) shared_outs(%[[INIT0:.+]] = %[[INIT]])
//  CHECK-DAG:     %[[TS_P:.+]] = affine.min #[[$MAP0]](%[[IV0]])[%[[P]]]
//  CHECK-DAG:     %[[TS_Q:.+]] = affine.min #[[$MAP1]](%[[IV1]])[%[[Q]]]
//  CHECK-DAG:     %[[TS_C:.+]] = affine.min #[[$MAP2]](%[[IV2]])[%[[C]]]
//  CHECK-DAG:     %[[TS_H:.+]] = affine.apply #[[$MAP3]](%[[TS_P]])[%[[R]]]
//  CHECK-DAG:     %[[TS_W:.+]] = affine.apply #[[$MAP4]](%[[TS_Q]])[%[[S]]]
//  CHECK-DAG:     %[[INPUT_TILE:.+]] = tensor.extract_slice %[[INPUT]]
// CHECK-SAME:         [0, %[[IV0]], %[[IV1]], %[[IV2]]] [%[[N]], %[[TS_H]], %[[TS_W]], %[[TS_C]]]
//  CHECK-DAG:     %[[FILTER_TILE:.+]] = tensor.extract_slice %[[FILTER]]
// CHECK-SAME:         [%[[IV0]], %[[IV1]], %[[IV2]], 0] [%[[TS_P]], %[[TS_Q]], %[[TS_C]], %[[F]]]
//  CHECK-DAG:     %[[INIT_TILE:.+]] = tensor.extract_slice %[[INIT0]]
// CHECK-SAME:         [0, 0, 0, 0] [%[[N]], %[[R]], %[[S]], %[[F]]]
//      CHECK:     %[[CONV_TILE:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:         dilation = dense<[4, 5]> : tensor<2xi64>, strides = dense<[2, 3]> : tensor<2xi64>
// CHECK-SAME:         ins(%[[INPUT_TILE]], %[[FILTER_TILE]] :
// CHECK-SAME:         outs(%[[INIT_TILE]] :
//      CHECK:     scf.forall.in_parallel
//      CHECK:       tensor.parallel_insert_slice %[[CONV_TILE]] into %[[INIT0]]
// CHECK-SAME:           [0, 0, 0, 0] [%[[N]], %[[R]], %[[S]], %[[F]]] [1, 1, 1, 1]
//      CHECK:   return %[[RESULT]]

// -----

// CHECK: #[[$MAP_ADD:.+]] = affine_map<(d0, d1) -> (d0 + d1)>

func.func @indexed_semantics(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // Check that we correctly amend "linalg.index" results.

  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0: tensor<?x?xf32>)
    outs(%arg1: tensor<?x?xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %1 = linalg.index 0 : index
    %2 = linalg.index 1 : index
    %3 = arith.addi %1, %2 : index
    %4 = arith.index_cast %3 : index to i64
    %5 = arith.uitofp %4 : i64 to f32
    %6 = arith.addf %5, %arg2 : f32
    linalg.yield %6 : f32
  } -> (tensor<?x?xf32>)
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.tile_using_forall %generic [10, 20]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: @indexed_semantics
//       CHECK: scf.forall (%[[I0:.+]], %[[I1:.+]]) =
//       CHECK:   %[[INDEX0:.+]] = linalg.index 0
//       CHECK:   %[[INDEX0_AMENDED:.+]] = affine.apply #[[$MAP_ADD]](%[[INDEX0]], %[[I0]])
//       CHECK:   %[[INDEX1:.+]] = linalg.index 1
//       CHECK:   %[[INDEX1_AMENDED:.+]] = affine.apply #[[$MAP_ADD]](%[[INDEX1]], %[[I1]])
//       CHECK:   arith.addi %[[INDEX0_AMENDED]], %[[INDEX1_AMENDED]]

// -----

func.func @interchange_matmul(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.tile_using_forall %matmul [10, 20] interchange = [1, 0] mapping = [#gpu.block<y>, #gpu.block<x>]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (20, -d0 + s0)>
//  CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0)[s0] -> (10, -d0 + s0)>
//      CHECK-LABEL: func.func @interchange_matmul(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[OUTER:[a-zA-Z0-9]+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]
// CHECK-SAME:       (0, 0) to (%[[N]], %[[M]]) step (20, 10)
// CHECK-SAME:       shared_outs(%[[INIT0:.+]] = %[[ARG2]])
//  CHECK-DAG:     %[[TS_N:.+]] = affine.min #[[$MAP0]](%[[IV0]])[%[[N]]]
//  CHECK-DAG:     %[[TS_M:.+]] = affine.min #[[$MAP2]](%[[IV1]])[%[[M]]]
//  CHECK-DAG:     %[[LHS_TILE:.+]] = tensor.extract_slice %[[ARG0]]
// CHECK-SAME:         [%[[IV1]], 0] [%[[TS_M]], %[[K]]] [1, 1]
//  CHECK-DAG:     %[[RHS_TILE:.+]] = tensor.extract_slice %[[ARG1]]
// CHECK-SAME:         [0, %[[IV0]]] [%[[K]], %[[TS_N]]] [1, 1]
//  CHECK-DAG:     %[[INIT_TILE:.+]] = tensor.extract_slice %[[INIT0]]
// CHECK-SAME:         [%[[IV1]], %[[IV0]]] [%[[TS_M]], %[[TS_N]]] [1, 1]
//      CHECK:     %[[GEMM_TILE:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS_TILE]], %[[RHS_TILE]] :
// CHECK-SAME:         outs(%[[INIT_TILE]] :
//      CHECK:     scf.forall.in_parallel {
//      CHECK:       tensor.parallel_insert_slice %[[GEMM_TILE]] into %[[INIT0]]
// CHECK-SAME:           [%[[IV1]], %[[IV0]]] [%[[TS_M]], %[[TS_N]]] [1, 1]
//      CHECK:     } {mapping = [#gpu.block<y>, #gpu.block<x>]}
//      CHECK:   return %[[OUTER]]

// -----

func.func @check_scalar_operation(%arg0 : tensor<f32>) -> tensor<f32> {
  %init = tensor.empty() : tensor<f32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>],
      iterator_types = []}      
      ins(%arg0 : tensor<f32>) outs(%init : tensor<f32>){
    ^bb0(%b0 : f32, %b1 : f32):
      %1 = arith.mulf %b0, %b0 : f32
      linalg.yield %1 : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a = transform.test.tile_using_forall %generic []
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func @check_scalar_operation
//   CHECK-NOT:   scf.for
//       CHECK:   linalg.generic

// -----

func.func @check_scalar_memref_operation(%arg0 : memref<f32>, %arg1 : memref<f32>){
  linalg.generic {
      indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>],
      iterator_types = []}      
      ins(%arg0 : memref<f32>) outs(%arg1 : memref<f32>){
    ^bb0(%b0 : f32, %b1 : f32):
      %1 = arith.mulf %b0, %b0 : f32
      linalg.yield %1 : f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a = transform.test.tile_using_forall %generic []
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func @check_scalar_memref_operation
//   CHECK-NOT:   scf.for
//       CHECK:   linalg.generic
