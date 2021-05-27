// RUN: mlir-opt %s -test-linalg-tensor-fusion-transform-patterns -canonicalize -cse --split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-linalg-tiled-loop-fusion-transform-patterns -canonicalize -cse  --split-input-file | FileCheck %s --check-prefix=TLOOP

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
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (32, d0 - d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (16, -d0 + s0)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0] -> (64, -d0 + s0)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1) -> (64, d0 - d1)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s0, 32, -d0 + s1)>

//      CHECK: func @matmul_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: tensor<?x?xf32>

//  CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[C32:.+]] = constant 32 : index
//  CHECK-DAG:   %[[C64:.+]] = constant 64 : index
//  CHECK-DAG:   %[[C16:.+]] = constant 16 : index
//  CHECK-DAG:   %[[M:.+]] = memref.dim %[[ARG0]], %[[C0]]
//      CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:[a-zA-Z0-9]+]] =
// CHECK-SAME:     %[[C0]] to %[[M]] step %[[C32]]
// CHECK-SAME:     iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<?x?xf32>) {
//      CHECK:     %[[M_2:.+]] = memref.dim %[[ARG6]], %[[C0]]
//      CHECK:     %[[TILE_M_2:.+]] = affine.min #[[MAP1]](%[[M_2]], %[[IV0]])
//      CHECK:     %[[N3:.+]] = memref.dim %[[ARG6]], %[[C1]]
//      CHECK:     %[[ST_ARG6:.+]] = subtensor %[[ARG6]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M_2]], %[[N3]]]
//      CHECK:     %[[TILE_M_3:.+]] = affine.min #[[MAP5]](%[[IV0]])[%[[M]], %[[M]]]
//      CHECK:     %[[N1:.+]] = memref.dim %[[ARG0]], %[[C1]]
//      CHECK:     %[[ST_ARG0:.+]] = subtensor %[[ARG0]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M_3]], %[[N1]]]
//      CHECK:     %[[M_3:.+]] = memref.dim %[[ARG2]], %[[C0]]
//      CHECK:     %[[TILE_M_4:.+]] = affine.min #[[MAP5]](%[[IV0]])[%[[M_3]], %[[M]]]
//      CHECK:     %[[N2_2:.+]] = memref.dim %[[ARG2]], %[[C1]]
//      CHECK:     %[[ST_ARG2:.+]] = subtensor %[[ARG2]][%[[IV0]], 0]
// CHECK-SAME:       [%[[TILE_M_4]], %[[N2_2]]]
//      CHECK:     %[[LHS:.+]] = linalg.matmul
// CHECK-SAME:       __internal_linalg_transform__ = "after_lhs_fusion_producer"
// CHECK-SAME:       ins(%[[ST_ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:       outs(%[[ST_ARG2]] : tensor<?x?xf32>)
//      CHECK:     %[[N2:.+]] = memref.dim %[[ARG1]], %[[C1]]
//      CHECK:     %[[N3_2:.+]] = memref.dim %[[ARG3]], %[[C1]]
//      CHECK:     %[[YIELD0:.+]] = scf.for %[[IV1:[a-zA-Z0-9]+]] =
// CHECK-SAME:       %[[C0]] to %[[N3_2]] step %[[C64]]
// CHECK-SAME:       iter_args(%[[ARG8:.+]] = %[[ST_ARG6]]) -> (tensor<?x?xf32>) {
//      CHECK:       %[[YIELD1:.+]] = scf.for %[[IV2:[a-zA-Z0-9]+]] =
// CHECK-SAME:         %[[C0]] to %[[N2]] step %[[C16]]
// CHECK-SAME:         iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<?x?xf32>) {
//      CHECK:         %[[TILE_N2:.+]] = affine.min #[[MAP2]](%[[IV2]])[%[[N2]]]
//      CHECK:         %[[ST_LHS:.+]] = subtensor %[[LHS]][0, %[[IV2]]]
// CHECK-SAME:           [%[[TILE_M_3]], %[[TILE_N2]]]
//      CHECK:         %[[N2_3:.+]] = memref.dim %[[ARG3]], %[[C0]]
//      CHECK:         %[[TILE_N2_2:.+]] = affine.min #[[MAP2]](%[[IV2]])[%[[N2_3]]]
//      CHECK:         %[[TILE_N3:.+]] = affine.min #[[MAP3]](%[[IV1]])[%[[N3_2]]]
//      CHECK:         %[[ST_ARG3:.+]] = subtensor %[[ARG3]][%[[IV2]], %[[IV1]]]
// CHECK-SAME:           [%[[TILE_N2_2]], %[[TILE_N3]]]
//      CHECK:         %[[M_4:.+]] = memref.dim %[[ARG10]], %[[C0]]
//      CHECK:         %[[N3_3:.+]] = memref.dim %[[ARG10]], %[[C1]]
//      CHECK:         %[[TILE_N3_2:.+]] = affine.min #[[MAP4]](%[[N3_3]], %[[IV1]])
//      CHECK:         %[[ST_ARG4:.+]] = subtensor %[[ARG10]][0, %[[IV1]]]
// CHECK-SAME:           [%[[M_4]], %[[TILE_N3_2]]]
//      CHECK:         %[[ST_RESULT:.+]] = linalg.matmul
// CHECK-SAME:           __internal_linalg_transform__ = "after_lhs_fusion"
// CHECK-SAME:           ins(%[[ST_LHS]], %[[ST_ARG3]]
// CHECK-SAME:             : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:           outs(%[[ST_ARG4]] : tensor<?x?xf32>)
//      CHECK:         %[[UPDATE1:.+]] = subtensor_insert %[[ST_RESULT]]
// CHECK-SAME:           into %[[ARG10]][0, %[[IV1]]] [%[[M_4]], %[[TILE_N3_2]]]
//      CHECK:         scf.yield %[[UPDATE1]]
//      CHECK:       }
//      CHECK:       scf.yield %[[YIELD1]]
//      CHECK:     }
//      CHECK:     %[[UPDATE0:.+]] = subtensor_insert %[[YIELD0]] into
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

// TLOOP-DAG:  %[[C32:.*]] = constant 32 : index
// TLOOP-DAG:  %[[C64:.*]] = constant 64 : index
// TLOOP-DAG:  %[[C16:.*]] = constant 16 : index
// TLOOP-DAG:  %[[C0:.*]] = constant 0 : index
// TLOOP-DAG:  %[[C1:.*]] = constant 1 : index

// TLOOP:  %[[DIM_A0:.*]] = memref.dim %[[A]], %[[C0]] : [[TY:.*]]

// TLOOP:  %[[ABC:.*]] = linalg.tiled_loop (%[[IV0:.*]]) = (%[[C0]]) 
// TLOOP-SAME: to (%[[DIM_A0]]) step (%[[C32]]) 
// TLOOP-SAME: ins (%[[C_:.*]] = %[[C]]: tensor<?x?xf32>,
// TLOOP-SAME:      %[[A_:.*]] = %[[A]]: tensor<?x?xf32>,
// TLOOP-SAME:      %[[B_:.*]] = %[[B]]: tensor<?x?xf32>,
// TLOOP-SAME:      %[[AB_INIT_:.*]] = %[[AB_INIT]]: tensor<?x?xf32>)
// TLOOP-SAME: outs (%[[ABC_INIT_:.*]] = %[[ABC_INIT]]: tensor<?x?xf32>) {

// TLOOP:    %[[ABC_INIT_SUB:.*]] = subtensor %[[ABC_INIT_]][%[[IV0]], 0]
// TLOOP:    %[[A_SUB:.*]] = subtensor %[[A_]][%[[IV0]], 0]
// TLOOP:    %[[AB_INIT_SUB:.*]] = subtensor %[[AB_INIT_]][%[[IV0]], 0]

// TLOOP:    %[[AB_SUB:.*]] = linalg.matmul
// TLOOP-SAME:  ins(%[[A_SUB]], %[[B_]] : {{.*}}) outs(%[[AB_INIT_SUB]]

// TLOOP:    %[[DIM_B_1:.*]] = memref.dim %[[B_]], %[[C1]] : [[TY]]
// TLOOP:    %[[DIM_C_1:.*]] = memref.dim %[[C_]], %[[C1]] : [[TY]]

// TLOOP:    %[[ABC_SUB_:.*]] = linalg.tiled_loop (%[[IV1:.*]], %[[IV2:.*]]) = 
// TLOOP-SAME: (%[[C0]], %[[C0]]) to (%[[DIM_C_1]], %[[DIM_B_1]])
// TLOOP-SAME: step (%[[C64]], %[[C16]]) 
// TLOOP-SAME: ins (%[[AB_SUB_:.*]] = %[[AB_SUB]]: [[TY]],
// TLOOP-SAME:      %[[C__:.*]] = %[[C_]]: [[TY]])
// TLOOP-SAME: outs (%[[ABC_INIT_SUB_:.*]] = %[[ABC_INIT_SUB]]: [[TY]])
// TLOOP-SAME: iterators["parallel", "reduction"] {

// TLOOP:      %[[AB_SUB_SUB:.*]] = subtensor %[[AB_SUB_]][0, %[[IV2]]]
// TLOOP:      %[[C__SUB:.*]] = subtensor %[[C__]][%[[IV2]], %[[IV1]]]
// TLOOP:      %[[ABS_INIT_SUB_SUB:.*]] = subtensor %[[ABC_INIT_SUB_]][0, %[[IV1]]]

// TLOOP:      %[[ABC_SUB_SUB:.*]] = linalg.matmul
// TLOOP-SAME:  ins(%[[AB_SUB_SUB]], %[[C__SUB]] : [[TY]], [[TY]])
// TLOOP-SAME:  outs(%[[ABS_INIT_SUB_SUB]] : [[TY]]) -> [[TY]]

// TLOOP:      %[[RES0:.*]] = subtensor_insert %[[ABC_SUB_SUB]]
// TLOOP-SAME:   into %[[ABC_INIT_SUB_]][0, %[[IV1]]]
// TLOOP:      linalg.yield %[[RES0]] : [[TY]]
// TLOOP:    }
// TLOOP:    %[[RES1:.*]] = subtensor_insert %[[ABC_SUB_]] into %[[ABC_INIT_]][%[[IV0]], 0]
// TLOOP:    linalg.yield %[[RES1]] : [[TY]]
// TLOOP:  }
// TLOOP:  return %[[ABC]] : [[TY]]

// -----

module {
  func @matmul_plus_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
                           %arg2: tensor<?x?xf32>) -> tensor<?x?xf32>{
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = memref.dim %arg2, %c0 : tensor<?x?xf32>
    %1 = memref.dim %arg2, %c1 : tensor<?x?xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %3 = memref.dim %2, %c0 : tensor<?x?xf32>
    %4 = memref.dim %2, %c1 : tensor<?x?xf32>
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
        %7 = addf %arg3, %arg4 : f32
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
//       CHECK:       %[[ST_ARG6:.+]] = subtensor %[[ARG6]][%[[IV0]], %[[IV1]]]
//       CHECK:       %[[ST_ARG0:.+]] = subtensor %[[ARG0]][%[[IV0]], 0]
//       CHECK:       %[[ST_ARG1:.+]] = subtensor %[[ARG1]][0, %[[IV1]]]
//       CHECK:       %[[ST_ARG2:.+]] = subtensor %[[ARG2]][%[[IV0]], %[[IV1]]]
//       CHECK:       %[[LHS:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[ST_ARG0]], %[[ST_ARG1]]
//  CHECK-SAME:           : tensor<?x?xf32>, tensor<?x?xf32>)
//  CHECK-SAME:         outs(%[[ST_ARG2]] : tensor<?x?xf32>)
//       CHECK:       %[[ST_RESULT:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[LHS]] : tensor<?x?xf32>)
//  CHECK-SAME:         outs(%[[ST_ARG6]] : tensor<?x?xf32>)
//       CHECK:       %[[UPDATE:.+]] = subtensor_insert %[[ST_RESULT]]
//  CHECK-SAME:         into %[[ARG6]][%[[IV0]], %[[IV1]]]
//       CHECK:       scf.yield %[[UPDATE]]
//       CHECK:     scf.yield %[[YIELD]]
//       CHECK:   return %[[RESULT]]

// TLOOP-LABEL: func @matmul_plus_matmul
// TLOOP-SAME:    %[[A:[a-zA-Z0-9_]+]]: tensor<?x?xf32>,
// TLOOP-SAME:    %[[B:[a-zA-Z0-9_]+]]: tensor<?x?xf32>,
// TLOOP-SAME:    %[[AB:[a-zA-Z0-9_]+]]: tensor<?x?xf32>

// TLOOP-DAG:  %[[C32:.*]] = constant 32 : index
// TLOOP-DAG:  %[[C64:.*]] = constant 64 : index
// TLOOP-DAG:  %[[C0:.*]] = constant 0 : index
// TLOOP-DAG:  %[[C1:.*]] = constant 1 : index

// TLOOP:  %[[DIM_A_0:.*]] = memref.dim %[[A]], %[[C0]] : [[TY:.*]]
// TLOOP:  %[[DIM_B_1:.*]] = memref.dim %[[B]], %[[C1]] : [[TY]]

// TLOOP:  %[[INIT:.*]] = linalg.init_tensor [%[[DIM_A_0]], %[[DIM_B_1]]]

// TLOOP:  %[[RESULT:.*]] = linalg.tiled_loop (%[[IV0:.*]], %[[IV1:.*]]) =
// TLOOP-SAME: (%[[C0]], %[[C0]]) to (%[[DIM_A_0]], %[[DIM_B_1]])
// TLOOP-SAME: step (%[[C32]], %[[C64]])
// TLOOP-SAME: ins (%[[A_:.*]] = %[[A]]: [[TY]],
// TLOOP-SAME:      %[[B_:.*]] = %[[B]]: [[TY]],
// TLOOP-SAME:      %[[AB_:.*]] = %[[AB]]: [[TY]])
// TLOOP-SAME: outs (%[[INIT_:.*]] = %[[INIT]]: [[TY]]) {

// TLOOP:    %[[INIT_SUB:.*]] = subtensor %[[INIT_]][%[[IV0]], %[[IV1]]]
// TLOOP:    %[[A_SUB:.*]] = subtensor %[[A_]][%[[IV0]], 0]
// TLOOP:    %[[B_SUB:.*]] = subtensor %[[B_]][0, %[[IV1]]]
// TLOOP:    %[[AB_SUB_INIT:.*]] = subtensor %[[AB_]][%[[IV0]], %[[IV1]]]

// TLOOP:    %[[AB_SUB:.*]] = linalg.matmul
// TLOOP-SAME:  ins(%[[A_SUB]], %[[B_SUB]] : [[TY]], [[TY]])
// TLOOP-SAME:  outs(%[[AB_SUB_INIT]] : [[TY]])

// TLOOP:    %[[DOUBLE_AB:.*]] = linalg.generic
// TLOOP-SAME:  ins(%[[AB_SUB]] : [[TY]]) outs(%[[INIT_SUB]] : [[TY]])

// TLOOP:    %[[RESULT_SUB:.*]] = subtensor_insert
// TLOOP-SAME:  %[[DOUBLE_AB:.*]] into %[[INIT_]][%[[IV0]], %[[IV1]]]

// TLOOP:    linalg.yield %[[RESULT_SUB]] : [[TY]]
// TLOOP:  }
// TLOOP:  return %[[RESULT]] : [[TY]]

// -----

module {
  func @matmul_out_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
                      %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = constant 0.0 : f32
    %0 = linalg.fill(%arg0, %c0) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
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
//       CHECK: %[[C0:.*]] = constant 0.0{{.*}} : f32
//   CHECK-NOT: fill
//       CHECK: scf.for %[[I:.*]]{{.*}}iter_args(%{{.*}} = %[[ARG0]]) -> (tensor<?x?xf32>) {
//       CHECK:   scf.for %[[J:.*]]
//       CHECK:     %[[ST:.*]] = subtensor %[[ARG0]]
//       CHECK:     %[[ST_FILL:.*]] = linalg.fill(%[[ST]], %[[C0]]) {__internal_linalg_transform__ = "after_out_fusion_producer"} : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
//       CHECK:     %[[ST_MM_RES:.*]] = scf.for %[[K:.*]]{{.*}}iter_args(%[[BB:.*]] = %[[ST_FILL]]) -> (tensor<?x?xf32>) {
//   CHECK-NOT:       fill
//       CHECK:       %[[ST_MM:.*]] = linalg.matmul {__internal_linalg_transform__ = "after_out_fusion"} ins(%{{.*}}, %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[BB]] : tensor<?x?xf32>) -> tensor<?x?xf32>
//       CHECK:       scf.yield %[[ST_MM]] : tensor<?x?xf32>
//       CHECK:     %[[MM:.*]] = subtensor_insert %[[ST_MM_RES]] into {{.*}}
//       CHECK:     scf.yield %[[MM]] : tensor<?x?xf32>


// TLOOP-LABEL: func @matmul_out_fusion(
// TLOOP-SAME:    %[[OUT:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// TLOOP-SAME:    %[[A:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// TLOOP-SAME:    %[[B:[a-zA-Z0-9_]+]]: tensor<?x?xf32>

// TLOOP-DAG:  %[[C0_F32:.*]] = constant 0.0
// TLOOP-DAG:  %[[C32:.*]] = constant 32 : index
// TLOOP-DAG:  %[[C64:.*]] = constant 64 : index
// TLOOP-DAG:  %[[C16:.*]] = constant 16 : index
// TLOOP-DAG:  %[[C0:.*]] = constant 0 : index
// TLOOP-DAG:  %[[C1:.*]] = constant 1 : index

// TLOOP:  %[[DIM_A_0:.*]] = memref.dim %[[A]], %[[C0]] : [[TY:.*]]
// TLOOP:  %[[DIM_B_1:.*]] = memref.dim %[[B]], %[[C1]] : [[TY]]

// TLOOP:  %[[AB:.*]] = linalg.tiled_loop (%[[I:.*]], %[[J:.*]]) = 
// TLOOP-SAME: (%[[C0]], %[[C0]]) to (%[[DIM_A_0]], %[[DIM_B_1]])
// TLOOP-SAME: step (%[[C32]], %[[C64]])
// TLOOP-SAME: ins (%[[A_:.*]] = %[[A]]: [[TY]],
// TLOOP-SAME:      %[[B_:.*]] = %[[B]]: [[TY]])
// TLOOP-SAME: outs (%[[OUT_:.*]] = %[[OUT]]: [[TY]]) {

// TLOOP:    %[[DIM_A__1:.*]] = memref.dim %[[A_]], %[[C1]] : [[TY]]
// TLOOP:    %[[A_SUB:.*]] = subtensor %[[A_]][%[[I]], 0]
// TLOOP:    %[[B_SUB:.*]] = subtensor %[[B_]][0, %[[J]]]
// TLOOP:    %[[OUT_SUB:.*]] = subtensor %[[OUT_]][%[[I]], %[[J]]]
// TLOOP:    %[[INIT_SUB:.*]] = linalg.fill(%[[OUT_SUB]], %[[C0_F32]])

// TLOOP:    %[[AB_SUB:.*]] = linalg.tiled_loop (%[[K:.*]]) = (%[[C0]]) 
// TLOOP-SAME: to (%[[DIM_A__1]]) step (%[[C16]])
// TLOOP-SAME: ins (%[[A_SUB_:.*]] = %[[A_SUB]]: [[TY]],
// TLOOP-SAME:      %[[B_SUB_:.*]] = %[[B_SUB]]: [[TY]])
// TLOOP-SAME: outs (%[[INIT_SUB_:.*]] = %[[INIT_SUB]]: [[TY]])
// TLOOP-SAME: iterators["reduction"] {

// TLOOP:      %[[A_SUB_SUB:.*]] = subtensor %[[A_SUB_]][0, %[[K]]]
// TLOOP:      %[[B_SUB_SUB:.*]] = subtensor %[[B_SUB_]][%[[K]], 0]

// TLOOP:      %[[AB_SUB_SUB:.*]] = linalg.matmul
// TLOOP-SAME:   ins(%[[A_SUB_SUB]], %[[B_SUB_SUB]] : [[TY]], [[TY]])
// TLOOP-SAME:   outs(%[[INIT_SUB_]] : [[TY]]) -> [[TY]]
// TLOOP:      linalg.yield %[[AB_SUB_SUB]] : [[TY]]
// TLOOP:    }
// TLOOP:    %[[SUB_RESULT:.*]] = subtensor_insert %[[AB_SUB]]
// TLOOP-SAME:  into %[[OUT_]][%[[I]], %[[J]]]
// TLOOP:    linalg.yield %[[SUB_RESULT]] : [[TY]]
// TLOOP:  }
// TLOOP:  return %[[AB]] : [[TY]]
