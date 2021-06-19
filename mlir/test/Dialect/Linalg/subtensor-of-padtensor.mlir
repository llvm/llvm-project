// RUN: mlir-opt %s -test-linalg-transform-patterns=test-swap-subtensor-padtensor -canonicalize  -split-input-file | FileCheck %s

// CHECK-LABEL: @static_data_only(
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<4x5xf32>
//       CHECK:   %[[RESULT:.*]] = subtensor %[[ARG0]][1, 2] [2, 1] [1, 1] : tensor<4x5xf32> to tensor<2x1xf32>
//       CHECK:   return %[[RESULT]]
func @static_data_only(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<2x1xf32> {
  %0 = linalg.pad_tensor %arg0 low[0, 0] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad : f32
    } : tensor<4x5xf32> to tensor<11x13xf32>
  %1 = subtensor %0[1, 2] [2, 1] [1, 1] : tensor<11x13xf32> to tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}

// -----

// CHECK-LABEL: @static_high_pad_only
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x5xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   linalg.pad_tensor
//   CHECK-NOT:   subtensor
//       CHECK:   %[[RESULT:.*]] = tensor.generate
//       CHECK:     tensor.yield %[[PAD]]
//       CHECK:   return %[[RESULT]] : tensor<2x4xf32>
func @static_high_pad_only(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<2x4xf32> {
  %0 = linalg.pad_tensor %arg0 low[0, 0] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad : f32
    } : tensor<4x5xf32> to tensor<11x13xf32>
  %1 = subtensor %0[4, 5] [2, 4] [1, 1] : tensor<11x13xf32> to tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: @static_mixed_data_high_pad
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x5xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   linalg.pad_tensor
//       CHECK:   %[[SUBTENSOR:.*]] = subtensor %[[ARG0]][2, 4] [2, 1] [1, 1] : tensor<4x5xf32> to tensor<2x1xf32>
//       CHECK:   %[[RESULT:.*]] = linalg.pad_tensor %[[SUBTENSOR]] low[0, 0] high[1, 3]
//       CHECK:     linalg.yield %[[PAD]]
//       CHECK:   return %[[RESULT]] : tensor<3x4xf32>
func @static_mixed_data_high_pad(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<3x4xf32> {
  %0 = linalg.pad_tensor %arg0 low[0, 0] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad : f32
    } : tensor<4x5xf32> to tensor<11x13xf32>
  %1 = subtensor %0[2, 4] [3, 4] [1, 1] : tensor<11x13xf32> to tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @dynamic_high_pad
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<?x5xf32>
//   CHECK-NOT:   linalg.pad_tensor
//       CHECK:   %[[C0:.*]] = constant 0 : index
//       CHECK:   memref.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[RESULT:.*]] = scf.if %{{.*}} -> (tensor<3x4xf32>) {
//       CHECK:     %[[GEN:.*]] = tensor.generate
//       CHECK:     scf.yield %[[GEN]]
//       CHECK:   } else {
//       CHECK:     %[[SUBTENSOR:.*]] = subtensor %[[ARG0]][%{{.*}}, 4] [%{{.*}}, 1] [1, 1] : tensor<?x5xf32> to tensor<?x1xf32>
//       CHECK:     %[[PADTENSOR:.*]] = linalg.pad_tensor %[[SUBTENSOR]] low[0, 0] high[%{{.*}}, 3]
//       CHECK:     %[[CAST:.*]] = tensor.cast %[[PADTENSOR]] : tensor<?x4xf32> to tensor<3x4xf32>
//       CHECK:     scf.yield %[[CAST]]
//       CHECK:   }
//       CHECK:   return %[[RESULT]]
func @dynamic_high_pad(%arg0 : tensor<?x5xf32>, %h1: index, %pad : f32) -> tensor<3x4xf32> {
  %0 = linalg.pad_tensor %arg0 low[0, 0] high[%h1, 8] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad : f32
    } : tensor<?x5xf32> to tensor<?x13xf32>
  %1 = subtensor %0[2, 4] [3, 4] [1, 1] : tensor<?x13xf32> to tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

