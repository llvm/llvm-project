// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=2,3,4" -split-input-file | FileCheck %s

// CHECK-LABEL: func @matmul_tensors(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
func @matmul_tensors(
  %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<?x?xf32>) {
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<?x?xf32>) {
//      CHECK:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<?x?xf32>) {
//      CHECK:       %[[sTA:.*]] = subtensor %[[TA]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTB:.*]] = subtensor %[[TB]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTC:.*]] = subtensor %[[TC2]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTD:.*]] = linalg.matmul ins(%[[sTA]], %[[sTB]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:                                  outs(%[[sTC]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
//      CHECK:       %[[TD:.*]] = subtensor_insert %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<?x?xf32> into tensor<?x?xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<?x?xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<?x?xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<?x?xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>

//      CHECK: return %[[TD0]] : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @generic_op_tensors(
  %arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %0 = memref.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = memref.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = memref.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = linalg.init_tensor [%0, %1, %2] : tensor<?x?x?xf32>
  %4 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d2, d1)>,
                      affine_map<(d0, d1, d2) -> (d2, d1, d0)>],
     iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs(%3 : tensor<?x?x?xf32>) {
    ^bb0(%arg2 : f32, %arg3: f32, %arg4: f32):
      %5 = addf %arg2, %arg3 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?x?xf32>
  return %4 : tensor<?x?x?xf32>
}

// CHECK-LABEL: func @generic_op_tensors
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor
//       CHECK:   %[[TD0:.+]] = scf.for %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[TC0:.+]] = %[[INIT]]) -> (tensor<?x?x?xf32>) {
//       CHECK:     %[[TD1:.+]] = scf.for %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[TC1:.+]] = %[[TC0]]) -> (tensor<?x?x?xf32>) {
//       CHECK:       %[[TD2:.+]] = scf.for %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[TC2:.+]] = %[[TC1]]) -> (tensor<?x?x?xf32>) {
//       CHECK:       %[[STARG0:.+]] = subtensor %[[ARG0]][{{.+}}] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
//       CHECK:       %[[STARG1:.+]] = subtensor %[[ARG1]][{{.+}}] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
//       CHECK:       %[[STARG2:.+]] = subtensor %[[TC2]][{{.+}}] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
//       CHECK:       %[[STRETURN:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[STARG0]], %[[STARG1]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
//  CHECK-SAME:         outs(%[[STARG2]] : tensor<?x?x?xf32>)
//       CHECK:       %[[TD:.+]] = subtensor_insert %[[STRETURN]] into %[[TC2]]
//       CHECK:       scf.yield %[[TD]]
//       CHECK:     }
//       CHECK:     scf.yield %[[TD2]]
//       CHECK:   }
//       CHECK:   scf.yield %[[TD1]]
//       CHECK: }
//       CHECK: return %[[TD0]]

// -----

func @indexed_generic_op_tensors(
  %arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %0 = memref.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = memref.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = memref.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = linalg.init_tensor [%0, %1, %2] : tensor<?x?x?xf32>
  %4 = linalg.indexed_generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d2, d1)>,
                      affine_map<(d0, d1, d2) -> (d2, d1, d0)>],
     iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs(%3 : tensor<?x?x?xf32>) {
    ^bb0(%arg2 : index, %arg3 : index, %arg4 : index, %arg5 : f32, %arg6: f32, %arg7: f32):
      %5 = addf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?x?xf32>
  return %4 : tensor<?x?x?xf32>
}

// CHECK-LABEL: func @indexed_generic_op_tensors
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor
//       CHECK:   %[[TD0:.+]] = scf.for %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[TC0:.+]] = %[[INIT]]) -> (tensor<?x?x?xf32>) {
//       CHECK:     %[[TD1:.+]] = scf.for %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[TC1:.+]] = %[[TC0]]) -> (tensor<?x?x?xf32>) {
//       CHECK:       %[[TD2:.+]] = scf.for %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[TC2:.+]] = %[[TC1]]) -> (tensor<?x?x?xf32>) {
//       CHECK:       %[[STARG0:.+]] = subtensor %[[ARG0]][{{.+}}] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
//       CHECK:       %[[STARG1:.+]] = subtensor %[[ARG1]][{{.+}}] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
//       CHECK:       %[[STARG2:.+]] = subtensor %[[TC2]][{{.+}}] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
//       CHECK:       %[[STRETURN:.+]] = linalg.indexed_generic
//  CHECK-SAME:         ins(%[[STARG0]], %[[STARG1]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
//  CHECK-SAME:         outs(%[[STARG2]] : tensor<?x?x?xf32>)
//       CHECK:       %[[TD:.+]] = subtensor_insert %[[STRETURN]] into %[[TC2]]
//       CHECK:       scf.yield %[[TD]]
//       CHECK:     }
//       CHECK:     scf.yield %[[TD2]]
//       CHECK:   }
//       CHECK:   scf.yield %[[TD1]]
//       CHECK: }
//       CHECK: return %[[TD0]]

// -----

func @fill_tensors(%arg0 : index, %arg1 : index, %arg2 : f32) -> tensor<?x?xf32> {
  %0 = linalg.init_tensor [%arg0, %arg1] : tensor<?x?xf32>
  %1 = linalg.fill(%0, %arg2) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//       CHECK: func @fill_tensors
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor
//       CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:[a-zA-z0-9_]+]]
//  CHECK-SAME:     iter_args(%[[ARG4:.+]] = %[[INIT]]) -> (tensor<?x?xf32>) {
//       CHECK:     %[[YIELD_1:.+]] = scf.for %[[IV1:[a-zA-Z0-9_]+]]
//  CHECK-SAME:       iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<?x?xf32>) {
//       CHECK:       %[[FILL_TILE:.+]] = subtensor %[[ARG6]][%[[IV0]], %[[IV1]]]
//       CHECK:       %[[RESULT_TILE:.+]] = linalg.fill(%[[FILL_TILE]], %{{.+}})
//       CHECK:       %[[YIELD_2:.+]] = subtensor_insert %[[RESULT_TILE]]
//  CHECK-SAME:         into %[[ARG6]][%[[IV0]], %[[IV1]]]
//       CHECK:       scf.yield %[[YIELD_2]]
//       CHECK:     }
//       CHECK:     scf.yield %[[YIELD_1]]
//       CHECK:   }
//       CHECK:   return %[[RESULT]]
