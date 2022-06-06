// RUN: mlir-opt --split-input-file --tosa-layerwise-constant-fold %s | FileCheck %s

// CHECK-LABEL: @transpose_fold
func.func @transpose_fold(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: return %arg0
  %0 = arith.constant dense<[0, 1]> : tensor<2xi32>
  %1 = "tosa.transpose"(%arg0, %0) { perms = [1, 0] }: (tensor<3x4xf32>, tensor<2xi32>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// CHECK-LABEL: @transpose_nofold
func.func @transpose_nofold(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // CHECK: "tosa.transpose"
  %0 = arith.constant dense<[1, 0]> : tensor<2xi32>
  %1 = "tosa.transpose"(%arg0, %0) { perms = [1, 0] }: (tensor<3x3xf32>, tensor<2xi32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}

// CHECK-LABEL: @transpose_nofold_shape
func.func @transpose_nofold_shape(%arg0: tensor<3x4xf32>) -> tensor<?x?xf32> {
  // CHECK: "tosa.transpose"
  %0 = arith.constant dense<[1, 0]> : tensor<2xi32>
  %1 = "tosa.transpose"(%arg0, %0) { perms = [1, 0] }: (tensor<3x4xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: @transpose_fold_splat
func.func @transpose_fold_splat() -> tensor<3x2xf32> {
  %input = "tosa.const"() {value = dense<4.0> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  //               CHECK: %[[CST:.+]] = "tosa.const"()
  // CHECK-SAME{LITERAL}: value = dense<4.000000e+00> : tensor<3x2xf32>
  %1 = "tosa.transpose"(%input, %perms) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x2xf32>
}

// CHECK-LABEL: @transpose_fold_2d_float
func.func @transpose_fold_2d_float() -> tensor<3x2xf32> {
  %input = "tosa.const"() {value = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  //               CHECK: %[[CST:.+]] = "tosa.const"()
  // CHECK-SAME{LITERAL}: value = dense<[[0.000000e+00, 3.000000e+00], [1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00]]> : tensor<3x2xf32>
  %1 = "tosa.transpose"(%input, %perms) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x2xf32>
}

// CHECK-LABEL: @transpose_fold_4d_int
func.func @transpose_fold_4d_int() -> tensor<3x1x4x2xi32> {
  %input = "tosa.const"() {value = dense<[[
    [[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]],
    [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
  ]]> : tensor<1x2x3x4xi32>} : () -> tensor<1x2x3x4xi32>
  %perms = "tosa.const"() {value = dense<[2, 0, 3, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
  //               CHECK: %[[CST:.+]] = "tosa.const"()
  // CHECK-SAME{LITERAL}: value = dense<[
  // CHECK-SAME{LITERAL}:   [[[0, 12], [1, 13], [2, 14], [3, 15]]],
  // CHECK-SAME{LITERAL}:   [[[4, 16], [5, 17], [6, 18], [7, 19]]],
  // CHECK-SAME{LITERAL}:   [[[8, 20], [9, 21], [10, 22], [11, 23]]]
  // CHECK-SAME{LITERAL}: ]>
  %1 = "tosa.transpose"(%input, %perms) : (tensor<1x2x3x4xi32>, tensor<4xi64>) -> tensor<3x1x4x2xi32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x1x4x2xi32>
}

// CHECK-LABEL: @transpose_nofold_non_cst_input
func.func @transpose_nofold_non_cst_input(%input: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: tosa.transpose
  %1 = "tosa.transpose"(%input, %perms) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// CHECK-LABEL: @transpose_nofold_non_cst_perms
func.func @transpose_nofold_non_cst_perms(%perms: tensor<2xi32>) -> tensor<3x2xf32> {
  %input = "tosa.const"() {value = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  // CHECK: tosa.transpose
  %1 = "tosa.transpose"(%input, %perms) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// CHECK-LABEL: @transpose_nofold_multi_users
func.func @transpose_nofold_multi_users() -> (tensor<3x2xf32>, tensor<2x3xf32>) {
  %input = "tosa.const"() {value = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: tosa.transpose
  %1 = "tosa.transpose"(%input, %perms) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %1, %input : tensor<3x2xf32>, tensor<2x3xf32>
}

// CHECK-LABEL: @transpose_nofold_quantized_types
func.func @transpose_nofold_quantized_types() -> tensor<1x1x16x1x!quant.uniform<i8<-127:127>:f32:3, {1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,2.100000e+00,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01}>> {
  %perms = "tosa.const"() {value = dense<[1, 2, 3, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
  %input = "tosa.const"() {value = dense<[[[[-127, 127, 127, -127, -127, -127, -127, -127, -127, 127, 127, 127, 127, 127, -127, 127]]]]> : tensor<1x1x1x16xi8>} : () -> tensor<1x1x1x16xi8>
  // CHECK: tosa.transpose
  %0 = "tosa.transpose"(%input, %perms) : (tensor<1x1x1x16xi8>, tensor<4xi32>) -> tensor<1x1x16x1x!quant.uniform<i8<-127:127>:f32:3, {1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,2.100000e+00,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01}>>
  return %0: tensor<1x1x16x1x!quant.uniform<i8<-127:127>:f32:3, {1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,2.100000e+00,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01}>>
}
