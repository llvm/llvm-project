// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(canonicalize{test-convergence}))' | FileCheck %s

// CHECK-LABEL: @dcast_fold
// CHECK-SAME: %[[ARG_0:.*]]: tensor

// CHECK: return %[[ARG_0]]

!qalias = !quant.uniform<u8:f32, 2.0:128>
func.func @dcast_fold(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = quant.qcast %arg0 : tensor<4xf32> to tensor<4x!qalias>
  %1 = quant.dcast %0 : tensor<4x!qalias> to tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @dcast_no_fold_source
// CHECK-SAME: %[[ARG_0:.*]]: tensor

// CHECK: %[[VAL_0:.*]] = quant.scast %[[ARG_0]]
// CHECK: %[[VAL_1:.*]] = quant.dcast %[[VAL_0]]
// CHECK: return %[[VAL_1]]

!qalias = !quant.uniform<u8:f32, 2.0:128>
func.func @dcast_no_fold_source(%arg0: tensor<4xi8>) -> tensor<4xf32> {
  %0 = quant.scast %arg0 : tensor<4xi8> to tensor<4x!qalias>
  %1 = quant.dcast %0 : tensor<4x!qalias> to tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @qcast_fold
// CHECK-SAME: %[[ARG_0:.*]]: tensor

// CHECK: return %[[ARG_0]]

!qalias = !quant.uniform<u8:f32, 2.0:128>
func.func @qcast_fold(%arg0: tensor<4x!qalias>) -> tensor<4x!qalias> {
  %0 = quant.dcast %arg0 : tensor<4x!qalias> to tensor<4xf32>
  %1 = quant.qcast %0 : tensor<4xf32> to tensor<4x!qalias>
  return %1 : tensor<4x!qalias>
}

// -----

// CHECK-LABEL: @qcast_no_fold_source
// CHECK-SAME: %[[ARG_0:.*]]: tensor

// CHECK: %[[VAL_0:.*]] = arith.negf %[[ARG_0]]
// CHECK: %[[VAL_1:.*]] = quant.qcast %[[VAL_0]]
// CHECK: return %[[VAL_1]]

!qalias = !quant.uniform<u8:f32, 2.0:128>
func.func @qcast_no_fold_source(%arg0: tensor<4xf32>) -> tensor<4x!qalias> {
  %0 = arith.negf %arg0 : tensor<4xf32>
  %1 = quant.qcast %0 : tensor<4xf32> to tensor<4x!qalias>
  return %1 : tensor<4x!qalias>
}

// -----

// CHECK-LABEL: @qcast_no_fold_type
// CHECK-SAME: %[[ARG_0:.*]]: tensor

// CHECK: %[[VAL_0:.*]] = quant.dcast %[[ARG_0]]
// CHECK: %[[VAL_1:.*]] = quant.qcast %[[VAL_0]]
// CHECK: return %[[VAL_1]]

!qalias = !quant.uniform<u8:f32, 2.0:128>
!qalias1 = !quant.uniform<u8:f32, 3.0:128>
func.func @qcast_no_fold_type(%arg0: tensor<4x!qalias>) -> tensor<4x!qalias1> {
  %0 = quant.dcast %arg0 : tensor<4x!qalias> to tensor<4xf32>
  %1 = quant.qcast %0 : tensor<4xf32> to tensor<4x!qalias1>
  return %1 : tensor<4x!qalias1>
}

// -----

// CHECK-LABEL: @scast_fold
// CHECK-SAME: %[[ARG_0:.*]]: tensor

// CHECK: return %[[ARG_0]]

!qalias = !quant.uniform<u8:f32, 2.0:128>
func.func @scast_fold(%arg0: tensor<4x!qalias>) -> tensor<4x!qalias> {
  %0 = quant.scast %arg0 : tensor<4x!qalias> to tensor<4xi8>
  %1 = quant.scast %0 : tensor<4xi8> to tensor<4x!qalias>
  return %1 : tensor<4x!qalias>
}

// -----

// CHECK-LABEL: @scast_no_fold_source
// CHECK-SAME: %[[ARG_0:.*]]: tensor

// CHECK: %[[QCAST:.*]] = quant.qcast %[[ARG_0]]
// CHECK: %[[SCAST:.*]] = quant.scast %[[QCAST]]
// CHECK: return %[[SCAST]]

!qalias = !quant.uniform<u8:f32, 2.0:128>
func.func @scast_no_fold_source(%arg0: tensor<4xf32>) -> tensor<4xi8> {
  %0 = quant.qcast %arg0 : tensor<4xf32> to tensor<4x!qalias>
  %1 = quant.scast %0 : tensor<4x!qalias> to tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// CHECK-LABEL: @scast_no_fold_type
// CHECK-SAME: %[[ARG_0:.*]]: tensor

// CHECK: %[[VAL_0:.*]] = quant.scast %[[ARG_0]]
// CHECK: %[[VAL_1:.*]] = quant.scast %[[VAL_0]]
// CHECK: return %[[VAL_1]]

!qalias = !quant.uniform<u8:f32, 2.0:128>
!qalias1 = !quant.uniform<u8:f32, 3.0:128>
func.func @scast_no_fold_type(%arg0: tensor<4x!qalias>) -> tensor<4x!qalias1> {
  %0 = quant.scast %arg0 : tensor<4x!qalias> to tensor<4xi8>
  %1 = quant.scast %0 : tensor<4xi8> to tensor<4x!qalias1>
  return %1 : tensor<4x!qalias1>
}

