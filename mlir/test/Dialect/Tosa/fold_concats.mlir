// RUN: mlir-opt --split-input-file --canonicalize %s | FileCheck %s

func.func @single_concat(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32> {
  %0 = tosa.concat %arg0, %arg0 {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  return %0 : tensor<1x2x7x7xf32>
}

// CHECK-LABEL:   func.func @single_concat(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// CHECK:           return %[[VAL_1]] : tensor<1x2x7x7xf32>
// CHECK:         }

// -----

func.func @concat_different_axis(%arg0: tensor<1x1x7x7xf32>) -> tensor<2x2x7x7xf32> {
  %0 = tosa.concat %arg0, %arg0 {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<2x2x7x7xf32>
  return %1 : tensor<2x2x7x7xf32>
}

// CHECK-LABEL:   func.func @concat_different_axis(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<2x2x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.concat %[[VAL_1]], %[[VAL_1]] {axis = 0 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<2x2x7x7xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x2x7x7xf32>
// CHECK:         }

// -----

func.func @fold_concats(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
  %tmp = tensor.empty() : tensor<1x1x7x7xf32>
  %0 = tosa.concat %arg0, %arg0 {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = tosa.concat %tmp, %0, %tmp {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x2x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
  return %1 : tensor<1x4x7x7xf32>
}

// CHECK-LABEL:   func.func @fold_concats(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x1x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.concat %[[VAL_1]], %[[VAL_0]], %[[VAL_0]], %[[VAL_1]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x4x7x7xf32>
// CHECK:         }

// -----

func.func @nested_fold(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x8x7x7xf32> {
  %tmp = tensor.empty() : tensor<1x1x7x7xf32>
  %0 = tosa.concat %arg0, %arg0 {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = tosa.concat %tmp, %0, %tmp {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x2x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
  %2 = tosa.concat %1, %1 {axis = 1 : i32} : (tensor<1x4x7x7xf32>, tensor<1x4x7x7xf32>) -> tensor<1x8x7x7xf32>
  return %2 : tensor<1x8x7x7xf32>
}

// CHECK-LABEL:   func.func @nested_fold(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x8x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x1x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.concat %[[VAL_1]], %[[VAL_0]], %[[VAL_0]], %[[VAL_1]], %[[VAL_1]], %[[VAL_0]], %[[VAL_0]], %[[VAL_1]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x8x7x7xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x8x7x7xf32>
// CHECK:         }

// -----

func.func @wide_fold(%arg0: tensor<1x1x7x7xf32>, %arg1: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
  %0 = tosa.concat %arg0, %arg0 {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = tosa.concat %arg1, %arg1 {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %2 = tosa.concat %0, %1 {axis = 1 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
  return %2 : tensor<1x4x7x7xf32>
}

// CHECK-LABEL:   func.func @wide_fold(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<1x1x7x7xf32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
// CHECK:           %[[VAL_2:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]], %[[VAL_1]], %[[VAL_1]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x4x7x7xf32>
// CHECK:         }

// -----

func.func @partially_foldable(%arg0: tensor<1x1x8x8xf32>, %arg1: tensor<1x2x4x8xf32>) -> tensor<1x4x8x8xf32> {
  %0 = tosa.concat %arg0, %arg0 {axis = 1 : i32} : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x2x8x8xf32>
  %1 = tosa.concat %arg1, %arg1 {axis = 2 : i32} : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x8x8xf32>
  %2 = tosa.concat %0, %1 {axis = 1 : i32} : (tensor<1x2x8x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x4x8x8xf32>
  return %2 : tensor<1x4x8x8xf32>
}

// CHECK-LABEL:   func.func @partially_foldable(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<1x1x8x8xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: tensor<1x2x4x8xf32>) -> tensor<1x4x8x8xf32> {
// CHECK:           %[[VAL_2:.*]] = tosa.concat %[[VAL_1]], %[[VAL_1]] {axis = 2 : i32} : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x8x8xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]], %[[VAL_2]] {axis = 1 : i32} : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x4x8x8xf32>
// CHECK:           return %[[VAL_3]] : tensor<1x4x8x8xf32>
// CHECK:         }

// -----

// CHECK-LABEL: test_fold_small_const_concat
func.func @test_fold_small_const_concat() -> tensor<6xi8> {
  // CHECK-DAG: %[[VAL_0:.*]] = "tosa.const"() <{value = dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi8>}> : () -> tensor<6xi8>
  // CHECK: return %[[VAL_0]] : tensor<6xi8>
  %0 = "tosa.const"() <{value = dense<[1, 2]> : tensor<2xi8>}> : () -> tensor<2xi8>
  %1 = "tosa.const"() <{value = dense<[3, 4, 5]> : tensor<3xi8>}> : () -> tensor<3xi8>
  %2 = "tosa.const"() <{value = dense<6> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = "tosa.concat"(%0, %1, %2) <{axis = 0 : i32}> : (tensor<2xi8>, tensor<3xi8>, tensor<1xi8>) -> tensor<6xi8>
  func.return %3 : tensor<6xi8>
}

// -----

// CHECK-LABEL: test_no_fold_small_const_concat_with_non_const
func.func @test_no_fold_small_const_concat_with_non_const(%arg0: tensor<2xi8>, %arg1: tensor<3xi8>, %arg2: tensor<1xi8>) -> tensor<6xi8> {
  // CHECK: %[[VAL_3:.*]] = tosa.concat %arg0, %arg1, %arg2 {axis = 0 : i32} : (tensor<2xi8>, tensor<3xi8>, tensor<1xi8>) -> tensor<6xi8>
  // CHECK: return %[[VAL_3]] : tensor<6xi8>
  %1 = "tosa.concat"(%arg0, %arg1, %arg2) <{axis = 0 : i32}> : (tensor<2xi8>, tensor<3xi8>, tensor<1xi8>) -> tensor<6xi8>
  func.return %1 : tensor<6xi8>
}

// -----

// CHECK-LABEL: test_no_fold_small_const_concat_with_higher_dim
func.func @test_no_fold_small_const_concat_with_higher_dim() -> tensor<7xi8> {
  // CHECK-DAG: %[[VAL_0:.*]] = "tosa.const"() <{value = dense<[1, 2, 3]> : tensor<3xi8>}> : () -> tensor<3xi8>
  // CHECK-DAG: %[[VAL_1:.*]] = "tosa.const"() <{value = dense<[4, 5, 6]> : tensor<3xi8>}> : () -> tensor<3xi8>
  // CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() <{value = dense<7> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[VAL_3:.*]] = tosa.concat %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] {axis = 0 : i32} : (tensor<3xi8>, tensor<3xi8>, tensor<1xi8>) -> tensor<7xi8>
  // CHECK: return %[[VAL_3]] : tensor<7xi8>
  %0 = "tosa.const"() <{value = dense<[1, 2, 3]> : tensor<3xi8>}> : () -> tensor<3xi8>
  %1 = "tosa.const"() <{value = dense<[4, 5, 6]> : tensor<3xi8>}> : () -> tensor<3xi8>
  %2 = "tosa.const"() <{value = dense<7> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = "tosa.concat"(%0, %1, %2) <{axis = 0 : i32}> : (tensor<3xi8>, tensor<3xi8>, tensor<1xi8>) -> tensor<7xi8>
  func.return %3 : tensor<7xi8>
}

// -----

// CHECK-LABEL: test_no_fold_small_const_concat_with_higher_rank
func.func @test_no_fold_small_const_concat_with_higher_rank() -> tensor<1x6xi8> {
  // CHECK-DAG: %[[VAL_0:.*]] = "tosa.const"() <{value = dense<{{\[\[}}1, 2]]> : tensor<1x2xi8>}> : () -> tensor<1x2xi8>
  // CHECK-DAG: %[[VAL_1:.*]] = "tosa.const"() <{value = dense<{{\[\[}}3, 4, 5]]> : tensor<1x3xi8>}> : () -> tensor<1x3xi8>
  // CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() <{value = dense<6> : tensor<1x1xi8>}> : () -> tensor<1x1xi8>
  // CHECK-DAG: %[[VAL_3:.*]] = tosa.concat %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] {axis = 1 : i32} : (tensor<1x2xi8>, tensor<1x3xi8>, tensor<1x1xi8>) -> tensor<1x6xi8>
  // CHECK: return %[[VAL_3]] : tensor<1x6xi8>
  %0 = "tosa.const"() <{value = dense<[[1, 2]]> : tensor<1x2xi8>}> : () -> tensor<1x2xi8>
  %1 = "tosa.const"() <{value = dense<[[3, 4, 5]]> : tensor<1x3xi8>}> : () -> tensor<1x3xi8>
  %2 = "tosa.const"() <{value = dense<[[6]]> : tensor<1x1xi8>}> : () -> tensor<1x1xi8>
  %3 = "tosa.concat"(%0, %1, %2) <{axis = 1 : i32}> : (tensor<1x2xi8>, tensor<1x3xi8>, tensor<1x1xi8>) -> tensor<1x6xi8>
  func.return %3 : tensor<1x6xi8>
}
