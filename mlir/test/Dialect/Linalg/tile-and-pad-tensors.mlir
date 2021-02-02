// RUN: mlir-opt %s -test-linalg-transform-patterns=test-tile-and-pad-pattern -canonicalize | FileCheck %s

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

// Dynamic op has been canonicalized away.
//  CHECK-NOT:       linalg.matmul {{.*}} tensor<?x?xf32>

// Padding injects static information.
//      CHECK:       %[[pA:.*]] = linalg.pad_tensor %[[sTA]] low[%c0, %c0] high[%{{.*}}, %{{.*}}]
//      CHECK:         : tensor<?x?xf32> to tensor<2x4xf32>
//      CHECK:       %[[pB:.*]] = linalg.pad_tensor %[[sTB]] low[%c0, %c0] high[%{{.*}}, %{{.*}}]
//      CHECK:         : tensor<?x?xf32> to tensor<4x3xf32>
//      CHECK:       %[[pC:.*]] = linalg.pad_tensor %[[sTC]] low[%c0, %c0] high[%{{.*}}, %{{.*}}]
//      CHECK:         : tensor<?x?xf32> to tensor<2x3xf32>
//      CHECK:       %[[pD:.*]] = linalg.matmul ins(%[[pA]], %[[pB]] : tensor<2x4xf32>, tensor<4x3xf32>)
// CHECK-SAME:                                  outs(%[[pC]] : tensor<2x3xf32>)  -> tensor<2x3xf32>
//      CHECK:       %[[sTD:.*]] = subtensor %[[pD]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : tensor<2x3xf32> to tensor<?x?xf32>
//      CHECK:       %[[TD:.*]] = subtensor_insert %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<?x?xf32> into tensor<?x?xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<?x?xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<?x?xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<?x?xf32>
  %0 = linalg.matmul {__internal_linalg_transform__ = "tile-and-pad"}
      ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
     outs(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>

//      CHECK: return %[[TD0]] : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
