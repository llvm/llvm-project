// RUN: mlir-opt %s -test-linalg-tensor-fusion-transform-patterns -resolve-shaped-type-result-dims -canonicalize -cse --split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-linalg-tiled-loop-fusion-transform-patterns -resolve-shaped-type-result-dims -canonicalize -cse  --split-input-file | FileCheck %s --check-prefix=TLOOP

module {
  func @matmul_fusion(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                      %AB_init: tensor<?x?xf32>, %C: tensor<?x?xf32>,
                      %ABC_init: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %AB = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%AB_init : tensor<?x?xf32>) -> tensor<?x?xf32>   // <MxN1> <N1xN2>
    %ABC = linalg.matmul {__internal_linalg_transform__ = "lhs_fusion"}
      ins(%AB, %C : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%ABC_init : tensor<?x?xf32>) -> tensor<?x?xf32>   // <MxN2> <N2xN3>
    return %ABC : tensor<?x?xf32>
  }
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (32, -d0 + s0)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (16, -d0 + s0)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0] -> (64, -d0 + s0)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s0, 32, -d0 + s1)>

//      CHECK: func @matmul_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: tensor<?x?xf32>

//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//  CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//      CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:[a-zA-Z0-9]+]] =
// CHECK-SAME:     %[[C0]] to %[[M]] step %[[C32]]
// CHECK-SAME:     iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<?x?xf32>) {
//      CHECK:     %[[TILE_M_2:.+]] = affine.min #[[MAP1]](%[[IV0]])[%[[M]]]
//      CHECK:     %[[N3:.+]] = tensor.dim %[[ARG6]], %[[C1]]
//      CHECK:     %[[ST_ARG6:.+]] = tensor.extract_slice %[[ARG6]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M_2]], %[[N3]]]
//      CHECK:     %[[TILE_M_3:.+]] = affine.min #[[MAP5]](%[[IV0]])[%[[M]], %[[M]]]
//      CHECK:     %[[N1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:     %[[ST_ARG0:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M_3]], %[[N1]]]
//      CHECK:     %[[N2_2:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//      CHECK:     %[[ST_ARG2:.+]] = tensor.extract_slice %[[ARG2]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M_3]], %[[N2_2]]]
//      CHECK:     %[[LHS:.+]] = linalg.matmul
// CHECK-SAME:       __internal_linalg_transform__ = "after_lhs_fusion_producer"
// CHECK-SAME:       ins(%[[ST_ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:       outs(%[[ST_ARG2]] : tensor<?x?xf32>)
//      CHECK:     %[[N2:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:     %[[N3_2:.+]] = tensor.dim %[[ARG3]], %[[C1]]
//      CHECK:     %[[YIELD0:.+]] = scf.for %[[IV1:[a-zA-Z0-9]+]] =
// CHECK-SAME:       %[[C0]] to %[[N3_2]] step %[[C64]]
// CHECK-SAME:       iter_args(%[[ARG8:.+]] = %[[ST_ARG6]]) -> (tensor<?x?xf32>) {
//      CHECK:       %[[YIELD1:.+]] = scf.for %[[IV2:[a-zA-Z0-9]+]] =
// CHECK-SAME:         %[[C0]] to %[[N2]] step %[[C16]]
// CHECK-SAME:         iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<?x?xf32>) {
//      CHECK:         %[[TILE_N2:.+]] = affine.min #[[MAP2]](%[[IV2]])[%[[N2]]]
//      CHECK:         %[[ST_LHS:.+]] = tensor.extract_slice %[[LHS]][0, %[[IV2]]]
// CHECK-SAME:           [%[[TILE_M_3]], %[[TILE_N2]]]
//      CHECK:         %[[TILE_N3:.+]] = affine.min #[[MAP3]](%[[IV1]])[%[[N3_2]]]
//      CHECK:         %[[ST_ARG3:.+]] = tensor.extract_slice %[[ARG3]][%[[IV2]], %[[IV1]]]
// CHECK-SAME:           [%[[TILE_N2]], %[[TILE_N3]]]
//      CHECK:         %[[M_4:.+]] = tensor.dim %[[ARG10]], %[[C0]]
//      CHECK:         %[[ST_ARG4:.+]] = tensor.extract_slice %[[ARG10]][0, %[[IV1]]]
// CHECK-SAME:           [%[[M_4]], %[[TILE_N3]]]
//      CHECK:         %[[ST_RESULT:.+]] = linalg.matmul
// CHECK-SAME:           __internal_linalg_transform__ = "after_lhs_fusion"
// CHECK-SAME:           ins(%[[ST_LHS]], %[[ST_ARG3]]
// CHECK-SAME:             : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:           outs(%[[ST_ARG4]] : tensor<?x?xf32>)
//      CHECK:         %[[UPDATE1:.+]] = tensor.insert_slice %[[ST_RESULT]]
// CHECK-SAME:           into %[[ARG10]][0, %[[IV1]]] [%[[M_4]], %[[TILE_N3]]]
//      CHECK:         scf.yield %[[UPDATE1]]
//      CHECK:       }
//      CHECK:       scf.yield %[[YIELD1]]
//      CHECK:     }
//      CHECK:     %[[UPDATE0:.+]] = tensor.insert_slice %[[YIELD0]] into
// CHECK-SAME:       %[[ARG6]][%[[IV0]], 0] [%[[TILE_M_2]], %[[N3]]]
//      CHECK:     scf.yield %[[UPDATE0]]
//      CHECK:   }
//      CHECK:   return %[[RESULT]]

// TLOOP-LABEL:  func @matmul_fusion(
// TLOOP-SAME: %[[A:[a-zA-Z0-9_]+]]: tensor<?x?xf32>,
// TLOOP-SAME: %[[B:[a-zA-Z0-9_]+]]: tensor<?x?xf32>,
// TLOOP-SAME: %[[AB_INIT:[a-zA-Z0-9_]+]]: tensor<?x?xf32>,
// TLOOP-SAME: %[[C:[a-zA-Z0-9_]+]]: tensor<?x?xf32>,
// TLOOP-SAME: %[[ABC_INIT:[a-zA-Z0-9_]+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {

// TLOOP-DAG:  %[[C32:.*]] = arith.constant 32 : index
// TLOOP-DAG:  %[[C64:.*]] = arith.constant 64 : index
// TLOOP-DAG:  %[[C16:.*]] = arith.constant 16 : index
// TLOOP-DAG:  %[[C0:.*]] = arith.constant 0 : index
// TLOOP-DAG:  %[[C1:.*]] = arith.constant 1 : index

// TLOOP:  %[[DIM_A0:.*]] = tensor.dim %[[A]], %[[C0]] : [[TY:.*]]

// TLOOP:  %[[ABC:.*]] = linalg.tiled_loop (%[[IV0:.*]]) = (%[[C0]]) 
// TLOOP-SAME: to (%[[DIM_A0]]) step (%[[C32]]) 
// TLOOP-SAME: ins (%[[C_:.*]] = %[[C]]: tensor<?x?xf32>,
// TLOOP-SAME:      %[[A_:.*]] = %[[A]]: tensor<?x?xf32>,
// TLOOP-SAME:      %[[B_:.*]] = %[[B]]: tensor<?x?xf32>,
// TLOOP-SAME:      %[[AB_INIT_:.*]] = %[[AB_INIT]]: tensor<?x?xf32>)
// TLOOP-SAME: outs (%[[ABC_INIT_:.*]] = %[[ABC_INIT]]: tensor<?x?xf32>) {

// TLOOP:    %[[ABC_INIT_SUB:.*]] = tensor.extract_slice %[[ABC_INIT_]][%[[IV0]], 0]
// TLOOP:    %[[A_SUB:.*]] = tensor.extract_slice %[[A_]][%[[IV0]], 0]
// TLOOP:    %[[AB_INIT_SUB:.*]] = tensor.extract_slice %[[AB_INIT_]][%[[IV0]], 0]

// TLOOP:    %[[AB_SUB:.*]] = linalg.matmul
// TLOOP-SAME:  ins(%[[A_SUB]], %[[B_]] : {{.*}}) outs(%[[AB_INIT_SUB]]

// TLOOP:    %[[DIM_B_1:.*]] = tensor.dim %[[B]], %[[C1]] : [[TY]]
// TLOOP:    %[[DIM_C_1:.*]] = tensor.dim %[[C]], %[[C1]] : [[TY]]

// TLOOP:    %[[ABC_SUB_:.*]] = linalg.tiled_loop (%[[IV1:.*]], %[[IV2:.*]]) = 
// TLOOP-SAME: (%[[C0]], %[[C0]]) to (%[[DIM_C_1]], %[[DIM_B_1]])
// TLOOP-SAME: step (%[[C64]], %[[C16]]) 
// TLOOP-SAME: ins (%[[AB_SUB_:.*]] = %[[AB_SUB]]: [[TY]],
// TLOOP-SAME:      %[[C__:.*]] = %[[C_]]: [[TY]])
// TLOOP-SAME: outs (%[[ABC_INIT_SUB_:.*]] = %[[ABC_INIT_SUB]]: [[TY]])
// TLOOP-SAME: iterators["parallel", "reduction"] {

// TLOOP:      %[[AB_SUB_SUB:.*]] = tensor.extract_slice %[[AB_SUB_]][0, %[[IV2]]]
// TLOOP:      %[[C__SUB:.*]] = tensor.extract_slice %[[C__]][%[[IV2]], %[[IV1]]]
// TLOOP:      %[[ABS_INIT_SUB_SUB:.*]] = tensor.extract_slice %[[ABC_INIT_SUB_]][0, %[[IV1]]]

// TLOOP:      %[[ABC_SUB_SUB:.*]] = linalg.matmul
// TLOOP-SAME:  ins(%[[AB_SUB_SUB]], %[[C__SUB]] : [[TY]], [[TY]])
// TLOOP-SAME:  outs(%[[ABS_INIT_SUB_SUB]] : [[TY]]) -> [[TY]]

// TLOOP:      %[[RES0:.*]] = tensor.insert_slice %[[ABC_SUB_SUB]]
// TLOOP-SAME:   into %[[ABC_INIT_SUB_]][0, %[[IV1]]]
// TLOOP:      linalg.yield %[[RES0]] : [[TY]]
// TLOOP:    }
// TLOOP:    %[[RES1:.*]] = tensor.insert_slice %[[ABC_SUB_]] into %[[ABC_INIT_]][%[[IV0]], 0]
// TLOOP:    linalg.yield %[[RES1]] : [[TY]]
// TLOOP:  }
// TLOOP:  return %[[ABC]] : [[TY]]

// -----

module {
  func @matmul_plus_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
                           %arg2: tensor<?x?xf32>) -> tensor<?x?xf32>{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = tensor.dim %arg2, %c0 : tensor<?x?xf32>
    %1 = tensor.dim %arg2, %c1 : tensor<?x?xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %3 = tensor.dim %2, %c0 : tensor<?x?xf32>
    %4 = tensor.dim %2, %c1 : tensor<?x?xf32>
    %5 = linalg.init_tensor [%3, %4] : tensor<?x?xf32>
    %6 = linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"],
       __internal_linalg_transform__ = "transpose_fusion"}
      ins(%2, %2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%5 : tensor<?x?xf32>) {
      ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32) :
        %7 = arith.addf %arg3, %arg4 : f32
        linalg.yield %7 : f32
      } -> tensor<?x?xf32>
    return %6 : tensor<?x?xf32>
  }
}
//       CHECK: func @matmul_plus_matmul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//       CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     iter_args(%[[ARG4:.+]] = %{{[a-zA-Z0-9_]+}})
//       CHECK:     %[[YIELD:.+]] = scf.for %[[IV1:[a-zA-Z0-9_]+]]
//  CHECK-SAME:       iter_args(%[[ARG6:.+]] = %[[ARG4]])
//       CHECK:       %[[ST_ARG6:.+]] = tensor.extract_slice %[[ARG6]][%[[IV0]], %[[IV1]]]
//       CHECK:       %[[ST_ARG0:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], 0]
//       CHECK:       %[[ST_ARG1:.+]] = tensor.extract_slice %[[ARG1]][0, %[[IV1]]]
//       CHECK:       %[[ST_ARG2:.+]] = tensor.extract_slice %[[ARG2]][%[[IV0]], %[[IV1]]]
//       CHECK:       %[[LHS:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[ST_ARG0]], %[[ST_ARG1]]
//  CHECK-SAME:           : tensor<?x?xf32>, tensor<?x?xf32>)
//  CHECK-SAME:         outs(%[[ST_ARG2]] : tensor<?x?xf32>)
//       CHECK:       %[[ST_RESULT:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[LHS]] : tensor<?x?xf32>)
//  CHECK-SAME:         outs(%[[ST_ARG6]] : tensor<?x?xf32>)
//       CHECK:       %[[UPDATE:.+]] = tensor.insert_slice %[[ST_RESULT]]
//  CHECK-SAME:         into %[[ARG6]][%[[IV0]], %[[IV1]]]
//       CHECK:       scf.yield %[[UPDATE]]
//       CHECK:     scf.yield %[[YIELD]]
//       CHECK:   return %[[RESULT]]

// TLOOP-LABEL: func @matmul_plus_matmul
// TLOOP-SAME:    %[[A:[a-zA-Z0-9_]+]]: tensor<?x?xf32>,
// TLOOP-SAME:    %[[B:[a-zA-Z0-9_]+]]: tensor<?x?xf32>,
// TLOOP-SAME:    %[[AB:[a-zA-Z0-9_]+]]: tensor<?x?xf32>

// TLOOP-DAG:  %[[C32:.*]] = arith.constant 32 : index
// TLOOP-DAG:  %[[C64:.*]] = arith.constant 64 : index
// TLOOP-DAG:  %[[C0:.*]] = arith.constant 0 : index
// TLOOP-DAG:  %[[C1:.*]] = arith.constant 1 : index

// TLOOP:  %[[DIM_A_0:.*]] = tensor.dim %[[A]], %[[C0]] : [[TY:.*]]
// TLOOP:  %[[DIM_B_1:.*]] = tensor.dim %[[B]], %[[C1]] : [[TY]]

// TLOOP:  %[[INIT:.*]] = linalg.init_tensor [%[[DIM_A_0]], %[[DIM_B_1]]]

// TLOOP:  %[[RESULT:.*]] = linalg.tiled_loop (%[[IV0:.*]], %[[IV1:.*]]) =
// TLOOP-SAME: (%[[C0]], %[[C0]]) to (%[[DIM_A_0]], %[[DIM_B_1]])
// TLOOP-SAME: step (%[[C32]], %[[C64]])
// TLOOP-SAME: ins (%[[A_:.*]] = %[[A]]: [[TY]],
// TLOOP-SAME:      %[[B_:.*]] = %[[B]]: [[TY]],
// TLOOP-SAME:      %[[AB_:.*]] = %[[AB]]: [[TY]])
// TLOOP-SAME: outs (%[[INIT_:.*]] = %[[INIT]]: [[TY]]) {

// TLOOP:    %[[INIT_SUB:.*]] = tensor.extract_slice %[[INIT_]][%[[IV0]], %[[IV1]]]
// TLOOP:    %[[A_SUB:.*]] = tensor.extract_slice %[[A_]][%[[IV0]], 0]
// TLOOP:    %[[B_SUB:.*]] = tensor.extract_slice %[[B_]][0, %[[IV1]]]
// TLOOP:    %[[AB_SUB_INIT:.*]] = tensor.extract_slice %[[AB_]][%[[IV0]], %[[IV1]]]

// TLOOP:    %[[AB_SUB:.*]] = linalg.matmul
// TLOOP-SAME:  ins(%[[A_SUB]], %[[B_SUB]] : [[TY]], [[TY]])
// TLOOP-SAME:  outs(%[[AB_SUB_INIT]] : [[TY]])

// TLOOP:    %[[DOUBLE_AB:.*]] = linalg.generic
// TLOOP-SAME:  ins(%[[AB_SUB]] : [[TY]]) outs(%[[INIT_SUB]] : [[TY]])

// TLOOP:    %[[RESULT_SUB:.*]] = tensor.insert_slice
// TLOOP-SAME:  %[[DOUBLE_AB:.*]] into %[[INIT_]][%[[IV0]], %[[IV1]]]

// TLOOP:    linalg.yield %[[RESULT_SUB]] : [[TY]]
// TLOOP:  }
// TLOOP:  return %[[RESULT]] : [[TY]]

// -----

module {
  func @matmul_out_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
                      %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0.0 : f32
    %0 = linalg.fill(%c0, %arg0) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
    %1 = linalg.matmul {__internal_linalg_transform__ = "out_fusion"}
      ins(%arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func @matmul_out_fusion(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//       CHECK: %[[C0:.*]] = arith.constant 0.0{{.*}} : f32
//   CHECK-NOT: fill
//       CHECK: scf.for %[[I:.*]]{{.*}}iter_args(%{{.*}} = %[[ARG0]]) -> (tensor<?x?xf32>) {
//       CHECK:   scf.for %[[J:.*]]
//       CHECK:     %[[ST:.*]] = tensor.extract_slice %[[ARG0]]
//       CHECK:     %[[ST_FILL:.*]] = linalg.fill(%[[C0]], %[[ST]]) {__internal_linalg_transform__ = "after_out_fusion_producer"} : f32, tensor<?x?xf32> -> tensor<?x?xf32>
//       CHECK:     %[[ST_MM_RES:.*]] = scf.for %[[K:.*]]{{.*}}iter_args(%[[BB:.*]] = %[[ST_FILL]]) -> (tensor<?x?xf32>) {
//   CHECK-NOT:       fill
//       CHECK:       %[[ST_FILL_SUB:.*]] = tensor.extract_slice %[[BB]][0, 0]
//       CHECK:       %[[ST_MM_SUB:.*]] = linalg.matmul {__internal_linalg_transform__ = "after_out_fusion"} ins(%{{.*}}, %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ST_FILL_SUB]] : tensor<?x?xf32>) -> tensor<?x?xf32>
//       CHECK:       %[[ST_MM:.*]] = tensor.insert_slice %[[ST_MM_SUB]] into %[[BB]]
//       CHECK:       scf.yield %[[ST_MM]] : tensor<?x?xf32>
//       CHECK:     %[[MM:.*]] = tensor.insert_slice %[[ST_MM_RES]] into {{.*}}
//       CHECK:     scf.yield %[[MM]] : tensor<?x?xf32>


// TLOOP-LABEL: func @matmul_out_fusion(
// TLOOP-SAME:    %[[OUT:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// TLOOP-SAME:    %[[A:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// TLOOP-SAME:    %[[B:[a-zA-Z0-9_]+]]: tensor<?x?xf32>

// TLOOP-DAG:  %[[C0_F32:.*]] = arith.constant 0.0
// TLOOP-DAG:  %[[C32:.*]] = arith.constant 32 : index
// TLOOP-DAG:  %[[C64:.*]] = arith.constant 64 : index
// TLOOP-DAG:  %[[C16:.*]] = arith.constant 16 : index
// TLOOP-DAG:  %[[C0:.*]] = arith.constant 0 : index
// TLOOP-DAG:  %[[C1:.*]] = arith.constant 1 : index

// TLOOP:  %[[DIM_A_0:.*]] = tensor.dim %[[A]], %[[C0]] : [[TY:.*]]
// TLOOP:  %[[DIM_B_1:.*]] = tensor.dim %[[B]], %[[C1]] : [[TY]]

// TLOOP:  %[[AB:.*]] = linalg.tiled_loop (%[[I:.*]], %[[J:.*]]) =
// TLOOP-SAME: (%[[C0]], %[[C0]]) to (%[[DIM_A_0]], %[[DIM_B_1]])
// TLOOP-SAME: step (%[[C32]], %[[C64]])
// TLOOP-SAME: ins (%[[A_:.*]] = %[[A]]: [[TY]],
// TLOOP-SAME:      %[[B_:.*]] = %[[B]]: [[TY]],
// TLOOP-SAME:      %[[C0_F32_:.*]] = %[[C0_F32]]
// TLOOP-SAME: outs (%[[OUT_:.*]] = %[[OUT]]: [[TY]]) {

// TLOOP:    %[[DIM_A__1:.*]] = tensor.dim %[[A]], %[[C1]] : [[TY]]
// TLOOP:    %[[A_SUB:.*]] = tensor.extract_slice %[[A_]][%[[I]], 0]
// TLOOP:    %[[B_SUB:.*]] = tensor.extract_slice %[[B_]][0, %[[J]]]
// TLOOP:    %[[OUT_SUB:.*]] = tensor.extract_slice %[[OUT_]][%[[I]], %[[J]]]
// TLOOP:    %[[INIT_SUB:.*]] = linalg.fill(%[[C0_F32_]], %[[OUT_SUB]])

// TLOOP:    %[[AB_SUB:.*]] = linalg.tiled_loop (%[[K:.*]]) = (%[[C0]])
// TLOOP-SAME: to (%[[DIM_A__1]]) step (%[[C16]])
// TLOOP-SAME: ins (%[[A_SUB_:.*]] = %[[A_SUB]]: [[TY]],
// TLOOP-SAME:      %[[B_SUB_:.*]] = %[[B_SUB]]: [[TY]])
// TLOOP-SAME: outs (%[[INIT_SUB_:.*]] = %[[INIT_SUB]]: [[TY]])
// TLOOP-SAME: iterators["reduction"] {

// TLOOP:      %[[A_SUB_SUB:.*]] = tensor.extract_slice %[[A_SUB_]][0, %[[K]]]
// TLOOP:      %[[B_SUB_SUB:.*]] = tensor.extract_slice %[[B_SUB_]][%[[K]], 0]
// TLOOP:      %[[INIT_SUB_SUB:.*]] = tensor.extract_slice %[[INIT_SUB_]][0, 0]

// TLOOP:      %[[AB_SUB_SUB:.*]] = linalg.matmul
// TLOOP-SAME:   ins(%[[A_SUB_SUB]], %[[B_SUB_SUB]] : [[TY]], [[TY]])
// TLOOP-SAME:   outs(%[[INIT_SUB_SUB]] : [[TY]]) -> [[TY]]
// TLOOP:      %[[AB_SUB_:.*]] = tensor.insert_slice %[[AB_SUB_SUB]] into %[[INIT_SUB_]]
// TLOOP:      linalg.yield %[[AB_SUB_]] : [[TY]]
// TLOOP:    }
// TLOOP:    %[[SUB_RESULT:.*]] = tensor.insert_slice %[[AB_SUB]]
// TLOOP-SAME:  into %[[OUT_]][%[[I]], %[[J]]]
// TLOOP:    linalg.yield %[[SUB_RESULT]] : [[TY]]
// TLOOP:  }
// TLOOP:  return %[[AB]] : [[TY]]

// -----

module {
  func @generic_plus_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
                      %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0.0 : f32
    %0 = linalg.generic {
      indexing_maps = [affine_map<(m, n) -> ()>, affine_map<(m, n) -> (m, n)>],
      iterator_types = ["parallel", "parallel"]}
     ins(%c0 : f32)
    outs(%arg0: tensor<?x?xf32>) {
      ^bb(%0: f32, %1: f32) :
        linalg.yield %0 : f32
    } -> tensor<?x?xf32>
    %1 = linalg.matmul {__internal_linalg_transform__ = "out_fusion"}
      ins(%arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
}

// TLOOP-LABEL: func @generic_plus_matmul(
// TLOOP-SAME:    %[[OUT:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// TLOOP-SAME:    %[[A:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// TLOOP-SAME:    %[[B:[a-zA-Z0-9_]+]]: tensor<?x?xf32>

// TLOOP-DAG:  %[[C0_F32:.*]] = arith.constant 0.0
// TLOOP-DAG:  %[[C32:.*]] = arith.constant 32 : index
// TLOOP-DAG:  %[[C64:.*]] = arith.constant 64 : index
// TLOOP-DAG:  %[[C16:.*]] = arith.constant 16 : index
// TLOOP-DAG:  %[[C0:.*]] = arith.constant 0 : index
// TLOOP-DAG:  %[[C1:.*]] = arith.constant 1 : index

// TLOOP:  %[[DIM_A_0:.*]] = tensor.dim %[[A]], %[[C0]] : [[TY:.*]]
// TLOOP:  %[[DIM_B_1:.*]] = tensor.dim %[[B]], %[[C1]] : [[TY]]

// TLOOP:  %[[AB:.*]] = linalg.tiled_loop (%[[I:.*]], %[[J:.*]]) =
// TLOOP-SAME: (%[[C0]], %[[C0]]) to (%[[DIM_A_0]], %[[DIM_B_1]])
// TLOOP-SAME: step (%[[C32]], %[[C64]])
// TLOOP-SAME: ins (%[[A_:.*]] = %[[A]]: [[TY]],
// TLOOP-SAME:      %[[B_:.*]] = %[[B]]: [[TY]],
// TLOOP-SAME:      %[[C0_F32_:.*]] = %[[C0_F32]]
// TLOOP-SAME: outs (%[[OUT_:.*]] = %[[OUT]]: [[TY]]) {

// TLOOP:    %[[DIM_A__1:.*]] = tensor.dim %[[A]], %[[C1]] : [[TY]]
// TLOOP:    %[[A_SUB:.*]] = tensor.extract_slice %[[A_]][%[[I]], 0]
// TLOOP:    %[[B_SUB:.*]] = tensor.extract_slice %[[B_]][0, %[[J]]]
// TLOOP:    %[[OUT_SUB:.*]] = tensor.extract_slice %[[OUT_]][%[[I]], %[[J]]]
// TLOOP:    %[[INIT_SUB:.*]] = linalg.generic
// TLOOP-SAME: ins(%[[C0_F32_]]
// TLOOP-SAME: outs(%[[OUT_SUB]]

// TLOOP:    %[[AB_SUB:.*]] = linalg.tiled_loop (%[[K:.*]]) = (%[[C0]])
// TLOOP-SAME: to (%[[DIM_A__1]]) step (%[[C16]])
// TLOOP-SAME: ins (%[[A_SUB_:.*]] = %[[A_SUB]]: [[TY]],
// TLOOP-SAME:      %[[B_SUB_:.*]] = %[[B_SUB]]: [[TY]])
// TLOOP-SAME: outs (%[[INIT_SUB_:.*]] = %[[INIT_SUB]]: [[TY]])
// TLOOP-SAME: iterators["reduction"] {

// TLOOP:      %[[A_SUB_SUB:.*]] = tensor.extract_slice %[[A_SUB_]][0, %[[K]]]
// TLOOP:      %[[B_SUB_SUB:.*]] = tensor.extract_slice %[[B_SUB_]][%[[K]], 0]
// TLOOP:      %[[INIT_SUB_SUB:.*]] = tensor.extract_slice %[[INIT_SUB_]][0, 0]

// TLOOP:      %[[AB_SUB_SUB:.*]] = linalg.matmul
// TLOOP-SAME:   ins(%[[A_SUB_SUB]], %[[B_SUB_SUB]] : [[TY]], [[TY]])
// TLOOP-SAME:   outs(%[[INIT_SUB_SUB]] : [[TY]]) -> [[TY]]
// TLOOP:      %[[AB_SUB_:.*]] = tensor.insert_slice %[[AB_SUB_SUB]] into %[[INIT_SUB_]]
// TLOOP:      linalg.yield %[[AB_SUB_]] : [[TY]]
// TLOOP:    }
// TLOOP:    %[[SUB_RESULT:.*]] = tensor.insert_slice %[[AB_SUB]]
// TLOOP-SAME:  into %[[OUT_]][%[[I]], %[[J]]]
// TLOOP:    linalg.yield %[[SUB_RESULT]] : [[TY]]
// TLOOP:  }
// TLOOP:  return %[[AB]] : [[TY]]

