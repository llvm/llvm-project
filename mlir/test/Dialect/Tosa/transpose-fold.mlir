// RUN: mlir-opt %s --canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @test_cancel_transpose_transpose(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<1x2x3xi32>) -> tensor<1x2x3xi32> {
// CHECK:           return %[[VAL_0]] : tensor<1x2x3xi32>
// CHECK:         }

func.func @test_cancel_transpose_transpose(%arg0: tensor<1x2x3xi32>) -> (tensor<1x2x3xi32>) {
	%0 = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
	%1 = tosa.transpose %arg0, %0 : (tensor<1x2x3xi32>, tensor<3xi32>) -> tensor<2x3x1xi32>
	%2 = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
	%3 = tosa.transpose %1, %2 : (tensor<2x3x1xi32>, tensor<3xi32>) -> tensor<1x2x3xi32>
  return %3 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL:   func.func @test_remove_identity_transpose(
// CHECK-SAME:                                              %[[VAL_0:.*]]: tensor<1x2x3xi32>) -> tensor<1x2x3xi32> {
// CHECK:           return %[[VAL_0]] : tensor<1x2x3xi32>
// CHECK:         }

func.func @test_remove_identity_transpose(%arg0: tensor<1x2x3xi32>) -> (tensor<1x2x3xi32>) {
	%0 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
	%1 = tosa.transpose %arg0, %0 : (tensor<1x2x3xi32>, tensor<3xi32>) -> tensor<1x2x3xi32>
  return %1 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL:   func.func @test_do_not_cancel_different_transpose(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: tensor<2x3x4x5xi32>) -> tensor<5x4x3x2xi32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<[3, 2, 1, 0]> : tensor<4xi32>
// CHECK:           %[[VAL_2:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_1]] : (tensor<2x3x4x5xi32>, tensor<4xi32>) -> tensor<5x4x3x2xi32>
// CHECK:           return %[[VAL_2]] : tensor<5x4x3x2xi32>
// CHECK:         }

func.func @test_do_not_cancel_different_transpose(%arg0: tensor<2x3x4x5xi32>) -> (tensor<5x4x3x2xi32>) {
	%0 = arith.constant dense<[1, 2, 0, 3]> : tensor<4xi32>
	%1 = tosa.transpose %arg0, %0 : (tensor<2x3x4x5xi32>, tensor<4xi32>) -> tensor<3x4x2x5xi32>
	%2 = arith.constant dense<[3, 1, 0, 2]> : tensor<4xi32>
	%3 = tosa.transpose %1, %2 : (tensor<3x4x2x5xi32>, tensor<4xi32>) -> tensor<5x4x3x2xi32>
  return %3 : tensor<5x4x3x2xi32>
}

// -----

// CHECK-LABEL:   func.func @test_prefer_compose_transpose(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: tensor<1x2x3x4xi32>) -> tensor<4x3x2x1xi32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<[3, 2, 1, 0]> : tensor<4xi32>
// CHECK:           %[[VAL_2:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_1]] : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<4x3x2x1xi32>
// CHECK:           return %[[VAL_2]] : tensor<4x3x2x1xi32>
// CHECK:         }

func.func @test_prefer_compose_transpose(%arg0: tensor<1x2x3x4xi32>) -> (tensor<4x3x2x1xi32>) {
	%0 = arith.constant dense<[1, 2, 0, 3]> : tensor<4xi32>
	%1 = tosa.transpose %arg0, %0 : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<2x3x1x4xi32>
	%2 = arith.constant dense<[3, 1, 0, 2]> : tensor<4xi32>
	%3 = tosa.transpose %1, %2 : (tensor<2x3x1x4xi32>, tensor<4xi32>) -> tensor<4x3x2x1xi32>
  return %3 : tensor<4x3x2x1xi32>
}
