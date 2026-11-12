// RUN: mlir-opt %s -test-linalg-transform-patterns=test-swap-subtensor-padtensor -canonicalize  -split-input-file | FileCheck %s

// CHECK-LABEL: @static_data_only(
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<4x5xf32>
//       CHECK:   %[[RESULT:.*]] = tensor.extract_slice %[[ARG0]][1, 2] [2, 1] [1, 1] : tensor<4x5xf32> to tensor<2x1xf32>
//       CHECK:   return %[[RESULT]]
func.func @static_data_only(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<2x1xf32> {
  %0 = tensor.pad %arg0 low[0, 0] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<4x5xf32> to tensor<11x13xf32>
  %1 = tensor.extract_slice %0[1, 2] [2, 1] [1, 1] : tensor<11x13xf32> to tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}

// -----

// CHECK-LABEL: @static_high_pad_only
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x5xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   tensor.pad
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   %[[RESULT:.*]] = tensor.generate
//       CHECK:     tensor.yield %[[PAD]]
//       CHECK:   return %[[RESULT]] : tensor<2x4xf32>
func.func @static_high_pad_only(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<2x4xf32> {
  %0 = tensor.pad %arg0 low[0, 0] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<4x5xf32> to tensor<11x13xf32>
  %1 = tensor.extract_slice %0[4, 5] [2, 4] [1, 1] : tensor<11x13xf32> to tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: @static_low_pad_only
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x5xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   tensor.pad
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   %[[RESULT:.*]] = tensor.generate
//       CHECK:     tensor.yield %[[PAD]]
//       CHECK:   return %[[RESULT]] : tensor<2x3xf32>
func.func @static_low_pad_only(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<2x3xf32> {
  %0 = tensor.pad %arg0 low[3, 7] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<4x5xf32> to tensor<14x20xf32>
  %1 = tensor.extract_slice %0[1, 3] [2, 3] [1, 1] : tensor<14x20xf32> to tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @static_low_pad_only_2
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x5xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   tensor.pad
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   %[[RESULT:.*]] = tensor.generate
//       CHECK:     tensor.yield %[[PAD]]
//       CHECK:   return %[[RESULT]] : tensor<1x3xf32>
func.func @static_low_pad_only_2(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<1x3xf32> {
  %0 = tensor.pad %arg0 low[3, 7] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<4x5xf32> to tensor<14x20xf32>
  %1 = tensor.extract_slice %0[1, 3] [1, 3] [1, 1] : tensor<14x20xf32> to tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}

// -----

// CHECK-LABEL: @static_mixed_data_high_pad
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x5xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   tensor.pad
//       CHECK:   %[[SUBTENSOR:.*]] = tensor.extract_slice %[[ARG0]][2, 4] [2, 1] [1, 1] : tensor<4x5xf32> to tensor<2x1xf32>
//       CHECK:   %[[RESULT:.*]] = tensor.pad %[[SUBTENSOR]] low[0, 0] high[1, 3]
//       CHECK:     tensor.yield %[[PAD]]
//       CHECK:   return %[[RESULT]] : tensor<3x4xf32>
func.func @static_mixed_data_high_pad(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<3x4xf32> {
  %0 = tensor.pad %arg0 low[0, 0] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<4x5xf32> to tensor<11x13xf32>
  %1 = tensor.extract_slice %0[2, 4] [3, 4] [1, 1] : tensor<11x13xf32> to tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @static_mixed_data_low_pad
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x5xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   tensor.pad
//       CHECK:   %[[SUBTENSOR:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [2, 1] [1, 1] : tensor<4x5xf32> to tensor<2x1xf32>
//       CHECK:   %[[RESULT:.*]] = tensor.pad %[[SUBTENSOR]] low[1, 3] high[0, 0]
//       CHECK:     tensor.yield %[[PAD]]
//       CHECK:   return %[[RESULT]] : tensor<3x4xf32>
func.func @static_mixed_data_low_pad(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<3x4xf32> {
  %0 = tensor.pad %arg0 low[3, 7] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<4x5xf32> to tensor<14x20xf32>
  %1 = tensor.extract_slice %0[2, 4] [3, 4] [1, 1] : tensor<14x20xf32> to tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @static_mixed_data_low_high_pad
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x5xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   tensor.pad
//       CHECK:   %[[RESULT:.*]] = tensor.pad %[[ARG0]] low[1, 1] high[2, 3]
//       CHECK:     tensor.yield %[[PAD]]
//       CHECK:   return %[[RESULT]] : tensor<7x9xf32>
func.func @static_mixed_data_low_high_pad(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<7x9xf32> {
  %0 = tensor.pad %arg0 low[2, 3] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<4x5xf32> to tensor<13x16xf32>
  %1 = tensor.extract_slice %0[1, 2] [7, 9] [1, 1] : tensor<13x16xf32> to tensor<7x9xf32>
  return %1 : tensor<7x9xf32>
}

// -----

// CHECK-LABEL: @static_rank_reduce
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<8x16x4xf32>, %[[PADVAL:.*]]: f32
//       CHECK:   %[[SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0, 0] [1, 14, 4] [1, 1, 1] : tensor<8x16x4xf32> to tensor<1x14x4xf32>
//       CHECK:   %[[PADDED:.*]] = tensor.pad %[[SLICE]] low[0, 2, 0] high[0, 0, 0] {
//       CHECK:   } : tensor<1x14x4xf32> to tensor<1x16x4xf32>
//       CHECK:   %[[RESULT:.*]] = tensor.extract_slice %[[PADDED]][0, 0, 0] [1, 16, 4] [1, 1, 1] : tensor<1x16x4xf32> to tensor<16x4xf32>
//       CHECK: return %[[RESULT]]
func.func @static_rank_reduce(%arg0: tensor<8x16x4xf32>, %pad: f32)
    -> tensor<16x4xf32> {
  %0 = tensor.pad %arg0 low[0, 2, 0] high[0, 0, 0] {
    ^bb0(%i: index, %j: index, %k: index):
      tensor.yield %pad : f32
  } : tensor<8x16x4xf32> to tensor<8x18x4xf32>
  %1 = tensor.extract_slice %0[0, 0, 0] [1, 16, 4] [1, 1, 1]
      : tensor<8x18x4xf32> to tensor<16x4xf32>
  return %1 : tensor<16x4xf32>
}

// -----

// CHECK-LABEL: @dynamic_high_pad
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<?x5xf32>
//   CHECK-NOT:   tensor.pad
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   tensor.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[RESULT:.*]] = scf.if %{{.*}} -> (tensor<3x4xf32>) {
//       CHECK:     %[[GEN:.*]] = tensor.generate
//       CHECK:     scf.yield %[[GEN]]
//       CHECK:   } else {
//       CHECK:     %[[SUBTENSOR:.*]] = tensor.extract_slice %[[ARG0]][%{{.*}}, 4] [%{{.*}}, 1] [1, 1] : tensor<?x5xf32> to tensor<?x1xf32>
//       CHECK:     %[[PADTENSOR:.*]] = tensor.pad %[[SUBTENSOR]] low[0, 0] high[%{{.*}}, 3]
//       CHECK:     scf.yield %[[PADTENSOR]]
//       CHECK:   }
//       CHECK:   return %[[RESULT]]
func.func @dynamic_high_pad(%arg0 : tensor<?x5xf32>, %h1: index, %pad : f32) -> tensor<3x4xf32> {
  %0 = tensor.pad %arg0 low[0, 0] high[%h1, 8] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<?x5xf32> to tensor<?x13xf32>
  %1 = tensor.extract_slice %0[2, 4] [3, 4] [1, 1] : tensor<?x13xf32> to tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @dynamic_extract_size
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<?x5xf32>, %[[ARG1:.*]]: index
//   CHECK-NOT:   tensor.pad
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   tensor.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[RESULT:.*]] = scf.if %{{.*}} -> (tensor<?x4xf32>) {
//       CHECK:     %[[GEN:.*]] = tensor.generate %[[ARG1]]
//       CHECK:     scf.yield %[[GEN]]
//       CHECK:   } else {
//       CHECK:     %[[SUBTENSOR:.*]] = tensor.extract_slice %[[ARG0]][%{{.*}}, 4] [%{{.*}}, 1] [1, 1] : tensor<?x5xf32> to tensor<?x1xf32>
//       CHECK:     %[[PADTENSOR:.*]] = tensor.pad %[[SUBTENSOR]] low[0, 0] high[%{{.*}}, 3]
//       CHECK:     scf.yield %[[PADTENSOR]]
//       CHECK:   }
//       CHECK:   return %[[RESULT]]
func.func @dynamic_extract_size(%arg0 : tensor<?x5xf32>, %s1: index, %pad : f32) -> tensor<?x4xf32> {
  %0 = tensor.pad %arg0 low[0, 0] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<?x5xf32> to tensor<?x13xf32>
  %1 = tensor.extract_slice %0[2, 4] [%s1, 4] [1, 1] : tensor<?x13xf32> to tensor<?x4xf32>
  return %1 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @dynamic_zero_low_padding
//       CHECK:   scf.if
//       CHECK:     tensor.generate
//       CHECK:   else
//       CHECK:     %[[SLICE:.*]] = tensor.extract_slice
//       CHECK:     tensor.pad %[[SLICE]] low[0, 0]
func.func @dynamic_zero_low_padding(%arg0 : tensor<?x?xf32>, %pad : f32,
                               %o1 : index, %o2 : index,
                               %s1 : index, %s2 : index)
    -> tensor<?x?xf32> {
  %0 = tensor.pad %arg0 low[0, 0] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = tensor.extract_slice %0[%o1, %o2] [%s1, %s2] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @dynamic_zero_high_padding
//       CHECK:   scf.if
//       CHECK:     tensor.generate
//       CHECK:   else
//       CHECK:     %[[SLICE:.*]] = tensor.extract_slice
//       CHECK:     tensor.pad %[[SLICE]] low[%{{.*}}, %{{.*}}] high[0, 0]
func.func @dynamic_zero_high_padding(%arg0 : tensor<?x?xf32>, %pad : f32,
                                %o1 : index, %o2 : index,
                                %s1 : index, %s2 : index)
    -> tensor<?x?xf32> {
  %0 = tensor.pad %arg0 low[7, 8] high[0, 0] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = tensor.extract_slice %0[%o1, %o2] [%s1, %s2] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @dynamic_rank_reduce
//       CHECK:   %[[TEMP:.*]] = scf.if %{{.*}} -> (tensor<1x4xf32>) {
//       CHECK:     tensor.generate
//       CHECK:   } else {
//       CHECK:     %[[SLICE:.*]] = tensor.extract_slice %{{.*}} : tensor<?x5xf32> to tensor<?x1xf32>
//       CHECK:     tensor.pad %[[SLICE]] low[0, 0] high[%{{.*}}, 3] {
//       CHECK:     } : tensor<?x1xf32> to tensor<1x4xf32>
//       CHECK:   }
//       CHECK:   %[[RESULT:.*]] = tensor.extract_slice %[[TEMP]]{{.*}} : tensor<1x4xf32> to tensor<4xf32>
//       CHECK:   return %[[RESULT]]
func.func @dynamic_rank_reduce(%arg0 : tensor<?x5xf32>, %s1: index, %pad : f32) -> tensor<4xf32> {
  %0 = tensor.pad %arg0 low[0, 0] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<?x5xf32> to tensor<?x13xf32>
  %1 = tensor.extract_slice %0[2, 4] [1, 4] [1, 1] : tensor<?x13xf32> to tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @nopaddim_with_dynamic_extract(
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<3x4x5xf32>
//  CHECK-SAME:     %[[ARG1:.*]]: f32
//  CHECK-SAME:     %[[ARG2:.*]]: index
//       CHECK:   %[[RESULT:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 1, 2] [%[[ARG2]], 2, 1] [1, 1, 1] : tensor<3x4x5xf32> to tensor<?x2x1xf32>
//       CHECK:   return %[[RESULT]]
func.func @nopaddim_with_dynamic_extract(%arg0 : tensor<3x4x5xf32>, %pad : f32, %index : index)
    -> tensor<?x2x1xf32> {
  %0 = tensor.pad %arg0 low[0, 0, 0] high[0, 7, 8] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %pad : f32
    } : tensor<3x4x5xf32> to tensor<3x11x13xf32>
  %1 = tensor.extract_slice %0[%index, 1, 2] [%index, 2, 1] [1, 1, 1] : tensor<3x11x13xf32> to tensor<?x2x1xf32>
  return %1 : tensor<?x2x1xf32>
}
