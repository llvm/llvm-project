// RUN: mlir-opt -test-tiling-interface=tile-consumer-and-fuse-producer-using-scf-for -split-input-file %s | FileCheck %s

func.func @gemm_fill_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm = linalg.matmul {__internal_linalg_transform__ = "fusion"}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %gemm : tensor<?x?xf32>
}
//      CHECK: func.func @gemm_fill_fusion(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor
//      CHECK:   scf.for %[[IV0:[a-zA-Z0-9]+]] =
// CHECK-SAME:       iter_args(%[[ITERARG0:.+]] = %[[INIT]])
//      CHECK:     scf.for %[[IV1:[a-zA-Z0-9]+]] =
// CHECK-SAME:         iter_args(%[[ITERARG1:.+]] = %[[ITERARG0]])
//  CHECK-DAG:       %[[LHS_TILE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], 0]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[IV1]]]
//  CHECK-DAG:       %[[INIT_TILE:.+]] = tensor.extract_slice %[[INIT]][%[[IV0]], %[[IV1]]]
//      CHECK:       %[[FILL_TILE:.+]] = linalg.fill
// CHECK-SAME:           outs(%[[INIT_TILE]] :
//      CHECK:       %[[GEMM_TILE:.+]] = linalg.matmul
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]] :
// CHECK-SAME:           outs(%[[FILL_TILE]] :
//      CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[GEMM_TILE]] into %[[ITERARG1]][%[[IV0]], %[[IV1]]]
//      CHECK        scf.yield %[[INSERT]]

// -----

func.func @gemm_generic_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  %generic = linalg.generic {
      __internal_linalg_transform__ = "fusion",
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%gemm, %arg2 : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %add = arith.addf %b0, %b1 : f32
      linalg.yield %add : f32 
  } -> tensor<?x?xf32>
  return %generic : tensor<?x?xf32> 
}
//      CHECK: func.func @gemm_generic_fusion(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?xf32>)
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor
//      CHECK:   scf.for %[[IV0:[a-zA-Z0-9]+]] =
// CHECK-SAME:       iter_args(%[[ITERARG0:.+]] = %[[INIT]])
//      CHECK:     scf.for %[[IV1:[a-zA-Z0-9]+]] =
// CHECK-SAME:         iter_args(%[[ITERARG1:.+]] = %[[ITERARG0]])
//  CHECK-DAG:       %[[LHS_TILE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], 0]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[IV1]]]
//  CHECK-DAG:       %[[INIT_TILE:.+]] = tensor.extract_slice %[[INIT]][%[[IV0]], %[[IV1]]]
//      CHECK:       %[[FILL_TILE:.+]] = linalg.fill
// CHECK-SAME:           outs(%[[INIT_TILE]] :
//      CHECK:       %[[GEMM_TILE:.+]] = linalg.matmul
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]] :
// CHECK-SAME:           outs(%[[FILL_TILE]] :
//  CHECK-DAG:       %[[BIAS_TILE:.+]] = tensor.extract_slice %[[ARG2]][%[[IV1]]]
//  CHECK-DAG:       %[[OUTS_TILE:.+]] = tensor.extract_slice %[[ITERARG1]][%[[IV0]], %[[IV1]]]
//      CHECK:       %[[GENERIC_TILE:.+]] = linalg.generic
// CHECK-SAME:           ins(%[[GEMM_TILE]], %[[BIAS_TILE]] :
// CHECK-SAME:           outs(%[[OUTS_TILE]] :
//      CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[GENERIC_TILE]] into %[[ITERARG1]][%[[IV0]], %[[IV1]]]
//      CHECK        scf.yield %[[INSERT]]

// -----

func.func @gemm_gemm_fusion(%lhs0 : tensor<?x?xf32>, %rhs0 : tensor<?x?xf32>, %rhs1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %lhs0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %rhs0, %c1 : tensor<?x?xf32>
  %init0 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %fill0 = linalg.fill ins(%cst : f32) outs(%init0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm0 = linalg.matmul
      ins(%lhs0, %rhs0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %d2 = tensor.dim %rhs1, %c1 : tensor<?x?xf32>
  %init1 = linalg.init_tensor [%d0, %d2] : tensor<?x?xf32>
  %fill1 = linalg.fill ins(%cst : f32) outs(%init1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm1 = linalg.matmul  {__internal_linalg_transform__ = "gemm_fusion"}
      ins(%gemm0, %rhs1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %gemm1 : tensor<?x?xf32>
}
//      CHECK: func.func @gemm_gemm_fusion(
// CHECK-SAME:     %[[LHS0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[RHS0:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[RHS1:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[LHS0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[RHS0]], %[[C1]]
//  CHECK-DAG:   %[[INIT0:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[RHS1]], %[[C1]]
//      CHECK:   %[[INIT1:.+]] = linalg.init_tensor [%[[D0]], %[[D2]]]
//      CHECK:   scf.for %[[IV:[a-zA-Z0-9]+]] =
// CHECK-SAME:       iter_args(%[[ITERARG:.+]] = %[[INIT1]])
//  CHECK-DAG:     %[[LHS0_TILE:.+]] = tensor.extract_slice %[[LHS0]][%[[IV]], 0]
//  CHECK-DAG:     %[[RHS0_TILE:.+]] = tensor.extract_slice %[[RHS0]][0, 0]
//  CHECK-DAG:     %[[INIT0_TILE:.+]] = tensor.extract_slice %[[INIT0]][%[[IV]], 0]
//      CHECK:     %[[FILL0_TILE:.+]] = linalg.fill
// CHECK-SAME:         outs(%[[INIT0_TILE]] :
//      CHECK:     %[[GEMM0_TILE:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS0_TILE]], %[[RHS0_TILE]] :
// CHECK-SAME:         outs(%[[FILL0_TILE]] :
//  CHECK-DAG:     %[[RHS1_TILE:.+]] = tensor.extract_slice %[[RHS1]][0, 0]
//  CHECK-DAG:     %[[INIT1_TILE:.+]] = tensor.extract_slice %[[INIT1]][%[[IV]], 0]
//      CHECK:     %[[FILL1_TILE:.+]] = linalg.fill
// CHECK-SAME:         outs(%[[INIT1_TILE]] :
//      CHECK:     %[[GEMM1_TILE:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[GEMM0_TILE]], %[[RHS1_TILE]] :
// CHECK-SAME:         outs(%[[FILL1_TILE]] :
//      CHECK:     %[[INSERT:.+]] = tensor.insert_slice %[[GEMM1_TILE]] into %[[ITERARG]][%[[IV]], 0]
//      CHECK      scf.yield %[[INSERT]]

// -----

func.func @gemm_transpose_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init0 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  %init1 = linalg.init_tensor [%d1, %d0] : tensor<?x?xf32>
  %transpose = linalg.generic {
      __internal_linalg_transform__ = "fusion",
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%gemm : tensor<?x?xf32>) outs(%init1 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      linalg.yield %b0 : f32 
  } -> tensor<?x?xf32>
  return %transpose : tensor<?x?xf32>
}
//      CHECK: func.func @gemm_transpose_fusion(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[INIT0:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]]
//  CHECK-DAG:   %[[INIT1:.+]] = linalg.init_tensor [%[[D1]], %[[D0]]]
//      CHECK:   scf.for %[[IV0:[a-zA-Z0-9]+]] =
// CHECK-SAME:       iter_args(%[[ITERARG0:.+]] = %[[INIT1]])
//      CHECK:     scf.for %[[IV1:[a-zA-Z0-9]+]] =
// CHECK-SAME:         iter_args(%[[ITERARG1:.+]] = %[[ITERARG0]])
//  CHECK-DAG:       %[[LHS_TILE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], 0]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[IV1]]]
//  CHECK-DAG:       %[[INIT0_TILE:.+]] = tensor.extract_slice %[[INIT0]][%[[IV0]], %[[IV1]]]
//      CHECK:       %[[FILL_TILE:.+]] = linalg.fill
// CHECK-SAME:           outs(%[[INIT0_TILE]] :
//      CHECK:       %[[GEMM_TILE:.+]] = linalg.matmul
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]] :
// CHECK-SAME:           outs(%[[FILL_TILE]] :
//  CHECK-DAG:       %[[OUTS_TILE:.+]] = tensor.extract_slice %[[ITERARG1]][%[[IV1]], %[[IV0]]]
//      CHECK:       %[[GENERIC_TILE:.+]] = linalg.generic
// CHECK-SAME:           ins(%[[GEMM_TILE]] :
// CHECK-SAME:           outs(%[[OUTS_TILE]] :
//      CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[GENERIC_TILE]] into %[[ITERARG1]][%[[IV1]], %[[IV0]]]
//      CHECK        scf.yield %[[INSERT]]

// -----

func.func @interchange_matmul_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %cst = arith.constant 0.0 : f32
  %0 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = linalg.generic {
      __internal_linalg_transform__ = "gemm_interchange_fusion",
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%2 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %4 = arith.addf %b0, %b0 : f32
        linalg.yield %4 : f32
      } -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}
//      CHECK: func.func @interchange_matmul_fusion(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor
//      CHECK:   scf.for %[[IV0:[a-zA-Z0-9]+]] =
// CHECK-SAME:       iter_args(%[[ITERARG0:.+]] = %[[INIT]])
//      CHECK:     scf.for %[[IV1:[a-zA-Z0-9]+]] =
// CHECK-SAME:         iter_args(%[[ITERARG1:.+]] = %[[ITERARG0]])
//  CHECK-DAG:       %[[LHS_TILE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV1]], 0]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[IV0]]]
//  CHECK-DAG:       %[[INIT_TILE:.+]] = tensor.extract_slice %[[INIT]][%[[IV1]], %[[IV0]]]
//      CHECK:       %[[FILL_TILE:.+]] = linalg.fill
// CHECK-SAME:           outs(%[[INIT_TILE]] :
//      CHECK:       %[[GEMM_TILE:.+]] = linalg.matmul
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]] :
// CHECK-SAME:           outs(%[[FILL_TILE]] :
//      CHECK:       %[[INIT_TILE_2:.+]] = tensor.extract_slice %[[ITERARG1]][%[[IV1]], %[[IV0]]]
//      CHECK:       %[[GENERIC_TILE:.+]] = linalg.generic
// CHECK-SAME:           ins(%[[GEMM_TILE]] :
// CHECK-SAME:           outs(%[[INIT_TILE_2]] :
//      CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[GENERIC_TILE]] into %[[ITERARG1]][%[[IV1]], %[[IV0]]]
//      CHECK        scf.yield %[[INSERT]]
