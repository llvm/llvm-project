// RUN: mlir-opt -test-tiling-interface=tile-consumer-fuse-and-yield-producer-using-scf-for -cse -split-input-file %s | FileCheck %s

func.func @gemm_gemm_fusion_yield_both(%lhs0 : tensor<?x?xf32>, %rhs0 : tensor<?x?xf32>, %rhs1 : tensor<?x?xf32>,
    %init0 : tensor<?x?xf32>, %init1 : tensor<?x?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %lhs0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %rhs0, %c1 : tensor<?x?xf32>
  %fill0 = linalg.fill ins(%cst : f32) outs(%init0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm0 = linalg.matmul
      ins(%lhs0, %rhs0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %d2 = tensor.dim %rhs1, %c1 : tensor<?x?xf32>
  %fill1 = linalg.fill ins(%cst : f32) outs(%init1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm1 = linalg.matmul  {__internal_linalg_transform__ = "gemm_sequence_fusion_and_yield"}
      ins(%gemm0, %rhs1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %gemm0, %gemm1 : tensor<?x?xf32>, tensor<?x?xf32>
}
//      CHECK: func.func @gemm_gemm_fusion_yield_both(
// CHECK-SAME:     %[[LHS0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[RHS0:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[RHS1:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[INIT0:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[INIT1:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[RESULT:.+]]:2 = scf.for %[[IV:[a-zA-Z0-9]+]] =
// CHECK-SAME:       iter_args(%[[ITERARG0:[a-zA-Z0-9]+]] = %[[INIT1]], %[[ITERARG1:[a-zA-Z0-9]+]] = %[[INIT0]])
//  CHECK-DAG:     %[[LHS0_TILE:.+]] = tensor.extract_slice %[[LHS0]][%[[IV]], 0]
//  CHECK-DAG:     %[[RHS0_TILE:.+]] = tensor.extract_slice %[[RHS0]][0, 0]
//  CHECK-DAG:     %[[INIT0_TILE:.+]] = tensor.extract_slice %[[ITERARG1]][%[[IV]], 0]
//      CHECK:     %[[FILL0_TILE:.+]] = linalg.fill
// CHECK-SAME:         outs(%[[INIT0_TILE]] :
//      CHECK:     %[[GEMM0_TILE:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS0_TILE]], %[[RHS0_TILE]] :
// CHECK-SAME:         outs(%[[FILL0_TILE]] :
//  CHECK-DAG:     %[[RHS1_TILE:.+]] = tensor.extract_slice %[[RHS1]][0, 0]
//  CHECK-DAG:     %[[INIT1_TILE:.+]] = tensor.extract_slice %[[ITERARG0]][%[[IV]], 0]
//      CHECK:     %[[FILL1_TILE:.+]] = linalg.fill
// CHECK-SAME:         outs(%[[INIT1_TILE]] :
//      CHECK:     %[[GEMM1_TILE:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[GEMM0_TILE]], %[[RHS1_TILE]] :
// CHECK-SAME:         outs(%[[FILL1_TILE]] :
//      CHECK:     %[[INSERT0:.+]] = tensor.insert_slice %[[GEMM1_TILE]] into %[[ITERARG0]][%[[IV]], 0]
//      CHECK:     %[[INSERT1:.+]] = tensor.insert_slice %[[GEMM0_TILE]] into %[[ITERARG1]][%[[IV]], 0]
//      CHECK:     scf.yield %[[INSERT0]], %[[INSERT1]]
