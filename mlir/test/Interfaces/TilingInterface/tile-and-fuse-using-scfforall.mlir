// RUN: mlir-opt --transform-interpreter --cse --split-input-file %s | FileCheck %s

func.func @gemm_fill_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %gemm : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_using_forall %matmul [10, 20]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @gemm_fill_fusion(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//      CHECK:   %[[INIT:.+]] = tensor.empty
//      CHECK:   scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) =
// CHECK-SAME:       shared_outs(%[[ITERARG0:.+]] = %[[INIT]])
//  CHECK-DAG:     %[[LHS_TILE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], 0]
//  CHECK-DAG:     %[[RHS_TILE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[IV1]]]
//  CHECK-DAG:     %[[INIT_TILE:.+]] = tensor.extract_slice %[[ITERARG0]][%[[IV0]], %[[IV1]]]
//      CHECK:     %[[FILL_TILE:.+]] = linalg.fill
// CHECK-SAME:         outs(%[[INIT_TILE]] :
//      CHECK:     %[[GEMM_TILE:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS_TILE]], %[[RHS_TILE]] :
// CHECK-SAME:         outs(%[[FILL_TILE]] :
//      CHECK:     scf.forall.in_parallel {
//      CHECK:       tensor.parallel_insert_slice %[[GEMM_TILE]] into %[[ITERARG0]][%[[IV0]], %[[IV1]]]
//      CHECK:     }

// -----

func.func @gemm_generic_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  %generic = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%gemm, %arg2 : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %add = arith.addf %b0, %b1 : f32
      linalg.yield %add : f32
  } -> tensor<?x?xf32>
  return %generic : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_using_forall %generic [10, 20]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @gemm_generic_fusion(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?xf32>)
//      CHECK:   %[[INIT:.+]] = tensor.empty
//      CHECK:   scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) =
// CHECK-SAME:       shared_outs(%[[ITERARG0:.+]] = %[[INIT]])
//  CHECK-DAG:     %[[LHS_TILE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], 0]
//  CHECK-DAG:     %[[RHS_TILE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[IV1]]]
//  CHECK-DAG:     %[[INIT_TILE:.+]] = tensor.extract_slice %[[INIT]][%[[IV0]], %[[IV1]]]
//      CHECK:     %[[FILL_TILE:.+]] = linalg.fill
// CHECK-SAME:         outs(%[[INIT_TILE]] :
//      CHECK:     %[[GEMM_TILE:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS_TILE]], %[[RHS_TILE]] :
// CHECK-SAME:         outs(%[[FILL_TILE]] :
//  CHECK-DAG:     %[[BIAS_TILE:.+]] = tensor.extract_slice %[[ARG2]][%[[IV1]]]
//  CHECK-DAG:     %[[OUTS_TILE:.+]] = tensor.extract_slice %[[ITERARG0]][%[[IV0]], %[[IV1]]]
//      CHECK:     %[[GENERIC_TILE:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[GEMM_TILE]], %[[BIAS_TILE]] :
// CHECK-SAME:         outs(%[[OUTS_TILE]] :
//      CHECK:     scf.forall.in_parallel {
//      CHECK:       tensor.parallel_insert_slice %[[GENERIC_TILE]] into %[[ITERARG0]][%[[IV0]], %[[IV1]]]
//      CHECK:     }

// -----

func.func @reduction_sequence(%arg0: tensor<30x3xf32>) -> tensor<30x3xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<30xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<30xf32>) -> tensor<30xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<30x3xf32>) outs(%1 : tensor<30xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %8 = arith.maximumf %arg2, %arg1 : f32
      linalg.yield %8 : f32
    } -> tensor<30xf32>
  %3 = tensor.empty() : tensor<30x3xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%0 : tensor<30xf32>) -> tensor<30xf32>
  %5:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0, %2 : tensor<30x3xf32>, tensor<30xf32>) outs(%4, %3 : tensor<30xf32>, tensor<30x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %8 = arith.subf %arg1, %arg2 : f32
      %9 = math.exp %8 : f32
      %10 = arith.addf %arg3, %9 : f32
      linalg.yield %10, %9 : f32, f32
    } -> (tensor<30xf32>, tensor<30x3xf32>)
  %6 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%5#1, %5#0 : tensor<30x3xf32>, tensor<30xf32>) outs(%3 : tensor<30x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %8 = arith.divf %arg1, %arg2 : f32
      linalg.yield %8 : f32
    } -> tensor<30x3xf32>
  return %6 : tensor<30x3xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %generics = transform.structured.match ops{["linalg.generic"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %generic1, %generic2, %generic3 = transform.split_handle %generics
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %a, %b = transform.test.fuse_using_forall %generic3 [10]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//       CHECK: func @reduction_sequence(%[[ARG0:.+]]: tensor<30x3xf32>)
//   CHECK-DAG:   %[[INIT0:.+]] = tensor.empty() : tensor<30xf32>
//   CHECK-DAG:   %[[INIT1:.+]] = tensor.empty() : tensor<30x3xf32>
//       CHECK:   %[[RESULT:[a-zA-Z0-9]+]] = scf.forall (%[[IV:[a-zA-Z0-9]+]])
//  CHECK-SAME:       shared_outs(%[[ITERARG0:[a-zA-Z0-9]+]] = %[[INIT1]])
//   CHECK-DAG:     %[[ARG0_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV]], 0]
//   CHECK-DAG:     %[[INIT0_SLICE:.+]] = tensor.extract_slice %[[INIT0]][%[[IV]]]
//       CHECK:     %[[FILL0:.+]] = linalg.fill
//  CHECK-SAME:         outs(%[[INIT0_SLICE]] :
//       CHECK:     %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[ARG0_SLICE]] :
//  CHECK-SAME:         outs(%[[FILL0]] :
//       CHECK:     %[[FILL1:.+]] = linalg.fill
//  CHECK-SAME:         outs(%[[INIT0_SLICE]] :
//       CHECK:     %[[INIT1_SLICE:.+]] = tensor.extract_slice %[[INIT1]][%[[IV]], 0]
//       CHECK:     %[[GENERIC1:.+]]:2 = linalg.generic
//  CHECK-SAME:         ins(%[[ARG0_SLICE]], %[[GENERIC0]] :
//  CHECK-SAME:         outs(%[[FILL1]], %[[INIT1_SLICE]] :
//       CHECK:     %[[ITERARG0_SLICE:.+]] = tensor.extract_slice %[[ITERARG0]][%[[IV]], 0]
//       CHECK:     %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC1]]#1, %[[GENERIC1]]#0 :
//  CHECK-SAME:         outs(%[[ITERARG0_SLICE]] :
//       CHECK:     scf.forall.in_parallel {
//       CHECK:       tensor.parallel_insert_slice %[[GENERIC2]] into %[[ITERARG0]][%[[IV]], 0]
//       CHECK:     }
//       CHECK:   return %[[RESULT]]
