// RUN: mlir-opt --split-input-file --canonicalize %s | FileCheck %s

func.func @single_concat(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  return %0 : tensor<1x2x7x7xf32>
}

// CHECK-LABEL:   func.func @single_concat(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.concat"(%[[VAL_0]], %[[VAL_0]]) <{axis = 1 : i64}> : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// CHECK:           return %[[VAL_1]] : tensor<1x2x7x7xf32>
// CHECK:         }

// -----

func.func @concat_different_axis(%arg0: tensor<1x1x7x7xf32>) -> tensor<2x2x7x7xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = "tosa.concat"(%0, %0) {axis = 0} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<2x2x7x7xf32>
  return %1 : tensor<2x2x7x7xf32>
}

// CHECK-LABEL:   func.func @concat_different_axis(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<2x2x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.concat"(%[[VAL_0]], %[[VAL_0]]) <{axis = 1 : i64}> : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = "tosa.concat"(%[[VAL_1]], %[[VAL_1]]) <{axis = 0 : i64}> : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<2x2x7x7xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x2x7x7xf32>
// CHECK:         }

// -----

func.func @fold_concats(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
  %tmp = tensor.empty() : tensor<1x1x7x7xf32>
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = "tosa.concat"(%tmp, %0, %tmp) {axis = 1} : (tensor<1x1x7x7xf32>, tensor<1x2x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
  return %1 : tensor<1x4x7x7xf32>
}

// CHECK-LABEL:   func.func @fold_concats(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x1x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = "tosa.concat"(%[[VAL_1]], %[[VAL_0]], %[[VAL_0]], %[[VAL_1]]) <{axis = 1 : i64}> : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x4x7x7xf32>
// CHECK:         }

// -----

func.func @nested_fold(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x8x7x7xf32> {
  %tmp = tensor.empty() : tensor<1x1x7x7xf32>
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = "tosa.concat"(%tmp, %0, %tmp) {axis = 1} : (tensor<1x1x7x7xf32>, tensor<1x2x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
  %2 = "tosa.concat"(%1, %1) {axis = 1} : (tensor<1x4x7x7xf32>, tensor<1x4x7x7xf32>) -> tensor<1x8x7x7xf32>
  return %2 : tensor<1x8x7x7xf32>
}

// CHECK-LABEL:   func.func @nested_fold(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x8x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x1x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = "tosa.concat"(%[[VAL_1]], %[[VAL_0]], %[[VAL_0]], %[[VAL_1]], %[[VAL_1]], %[[VAL_0]], %[[VAL_0]], %[[VAL_1]]) <{axis = 1 : i64}> : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x8x7x7xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x8x7x7xf32>
// CHECK:         }

// -----

func.func @wide_fold(%arg0: tensor<1x1x7x7xf32>, %arg1: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = "tosa.concat"(%arg1, %arg1) {axis = 1} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %2 = "tosa.concat"(%0, %1) {axis = 1} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
  return %2 : tensor<1x4x7x7xf32>
}

// CHECK-LABEL:   func.func @wide_fold(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<1x1x7x7xf32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.concat"(%[[VAL_0]], %[[VAL_0]], %[[VAL_1]], %[[VAL_1]]) <{axis = 1 : i64}> : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x4x7x7xf32>
// CHECK:         }

// -----

func.func @partially_foldable(%arg0: tensor<1x1x8x8xf32>, %arg1: tensor<1x2x4x8xf32>) -> tensor<1x4x8x8xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1} : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x2x8x8xf32>
  %1 = "tosa.concat"(%arg1, %arg1) {axis = 2} : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x8x8xf32>
  %2 = "tosa.concat"(%0, %1) {axis = 1} : (tensor<1x2x8x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x4x8x8xf32>
  return %2 : tensor<1x4x8x8xf32>
}

// CHECK-LABEL:   func.func @partially_foldable(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<1x1x8x8xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: tensor<1x2x4x8xf32>) -> tensor<1x4x8x8xf32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.concat"(%[[VAL_1]], %[[VAL_1]]) <{axis = 2 : i64}> : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x8x8xf32>
// CHECK:           %[[VAL_3:.*]] = "tosa.concat"(%[[VAL_0]], %[[VAL_0]], %[[VAL_2]]) <{axis = 1 : i64}> : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x4x8x8xf32>
// CHECK:           return %[[VAL_3]] : tensor<1x4x8x8xf32>
// CHECK:         }
