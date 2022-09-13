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

// -----

// CHECK-LABEL: @fold_add_zero_rhs_f32
func.func @fold_add_zero_rhs_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %zero = "tosa.const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %add = "tosa.add"(%arg0, %zero) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %arg0
  return %add : tensor<f32>
}

// -----

// CHECK-LABEL: @fold_add_zero_lhs_f32
func.func @fold_add_zero_lhs_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %zero = "tosa.const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %add = "tosa.add"(%zero, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %arg0
  return %add : tensor<f32>
}

// -----

// CHECK-LABEL: @fold_add_zero_rhs_i32
func.func @fold_add_zero_rhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %zero = "tosa.const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %add = "tosa.add"(%arg0, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %arg0
  return %add : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_add_zero_lhs_i32
func.func @fold_add_zero_lhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %zero = "tosa.const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %add = "tosa.add"(%zero, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %arg0
  return %add : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_add_splat_i32
func.func @fold_add_splat_i32() -> tensor<10xi32> {
  %one = "tosa.const"() {value = dense<1> : tensor<10xi32>} : () -> tensor<10xi32>
  %two = "tosa.const"() {value = dense<2> : tensor<10xi32>} : () -> tensor<10xi32>
  %add = "tosa.add"(%one, %two) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
  // CHECK: %[[THREE:.+]] = "tosa.const"() {value = dense<3> : tensor<10xi32>}
  // CHECK: return %[[THREE]]
  return %add : tensor<10xi32>
}

// -----

// CHECK-LABEL: @fold_add_splat_f32
func.func @fold_add_splat_f32() -> tensor<10xf32> {
  %one = "tosa.const"() {value = dense<1.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %two = "tosa.const"() {value = dense<2.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %add = "tosa.add"(%one, %two) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  // CHECK: %[[THREE:.+]] = "tosa.const"() {value = dense<3.000000e+00> : tensor<10xf32>}
  // CHECK: return %[[THREE]]
  return %add : tensor<10xf32>
}

// -----

// CHECK-LABEL: @fold_div_zero_lhs_i32
func.func @fold_div_zero_lhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %zero = "tosa.const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[ZERO:.+]] = "tosa.const"() {value = dense<0> : tensor<i32>}
  %div = "tosa.div"(%zero, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %[[ZERO]]
  return %div : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_div_one_rhs_i32
func.func @fold_div_one_rhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %one = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %div = "tosa.div"(%arg0, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %arg0
  return %div : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_div_splat_i32
func.func @fold_div_splat_i32() -> tensor<i32> {
  %lhs = "tosa.const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %rhs = "tosa.const"() {value = dense<-3> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() {value = dense<-3> : tensor<i32>}
  %div = "tosa.div"(%lhs, %rhs) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %[[SPLAT]]
  return %div : tensor<i32>
}

// -----


// CHECK-LABEL: @fold_mul_zero_rhs_f32
func.func @fold_mul_zero_rhs_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %zero = "tosa.const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[ZERO:.+]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>}
  %mul = "tosa.mul"(%arg0, %zero) {shift = 0 : i32} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %[[ZERO]]
  return %mul : tensor<f32>
}

// -----

// CHECK-LABEL: @fold_mul_zero_lhs_f32
func.func @fold_mul_zero_lhs_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %zero = "tosa.const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[ZERO:.+]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>}
  %mul = "tosa.mul"(%zero, %arg0) {shift = 0 : i32} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %[[ZERO]]
  return %mul : tensor<f32>
}

// -----

// CHECK-LABEL: @fold_mul_zero_rhs_i32
func.func @fold_mul_zero_rhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %zero = "tosa.const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[ZERO:.+]] = "tosa.const"() {value = dense<0> : tensor<i32>}
  %mul = "tosa.mul"(%arg0, %zero) {shift = 0 : i32} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %[[ZERO]]
  return %mul : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_mul_zero_lhs_i32
func.func @fold_mul_zero_lhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %zero = "tosa.const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[ZERO:.+]] = "tosa.const"() {value = dense<0> : tensor<i32>}
  %mul = "tosa.mul"(%zero, %arg0) {shift = 0 : i32} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %[[ZERO]]
  return %mul : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_mul_one_rhs_f32
func.func @fold_mul_one_rhs_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %one = "tosa.const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %mul = "tosa.mul"(%arg0, %one) {shift = 0 : i32} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %arg0
  return %mul : tensor<f32>
}

// -----

// CHECK-LABEL: @fold_mul_one_lhs_f32
func.func @fold_mul_one_lhs_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %one = "tosa.const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %mul = "tosa.mul"(%one, %arg0) {shift = 0 : i32} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %arg0
  return %mul : tensor<f32>
}

// -----

// CHECK-LABEL: @fold_mul_one_rhs_i32
func.func @fold_mul_one_rhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %one = "tosa.const"() {value = dense<64> : tensor<i32>} : () -> tensor<i32>
  %mul = "tosa.mul"(%arg0, %one) {shift = 6 : i32} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %arg0
  return %mul : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_mul_one_lhs_i32
func.func @fold_mul_one_lhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %one = "tosa.const"() {value = dense<64> : tensor<i32>} : () -> tensor<i32>
  %mul = "tosa.mul"(%one, %arg0) {shift = 6 : i32} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %arg0
  return %mul : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_mul_splat_i8
func.func @fold_mul_splat_i8() -> tensor<10xi8> {
  %one = "tosa.const"() {value = dense<17> : tensor<10xi8>} : () -> tensor<10xi8>
  %two = "tosa.const"() {value = dense<32> : tensor<10xi8>} : () -> tensor<10xi8>
  %mul = "tosa.mul"(%one, %two) {shift = 3 : i32} : (tensor<10xi8>, tensor<10xi8>) -> tensor<10xi8>
  // CHECK: %[[THREE:.+]] = "tosa.const"() {value = dense<68> : tensor<10xi8>}
  // CHECK: return %[[THREE]]
  return %mul : tensor<10xi8>
}

// -----

// CHECK-LABEL: @fold_mul_splat_f32
func.func @fold_mul_splat_f32() -> tensor<10xf32> {
  %one = "tosa.const"() {value = dense<3.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %two = "tosa.const"() {value = dense<2.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %mul = "tosa.mul"(%one, %two) {shift = 0 : i32} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  // CHECK: %[[THREE:.+]] = "tosa.const"() {value = dense<6.000000e+00> : tensor<10xf32>}
  // CHECK: return %[[THREE]]
  return %mul : tensor<10xf32>
}

// -----

// CHECK-LABEL: @fold_sub_zero_rhs_f32
func.func @fold_sub_zero_rhs_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %zero = "tosa.const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %sub = "tosa.sub"(%arg0, %zero) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %arg0
  return %sub : tensor<f32>
}

// -----

// CHECK-LABEL: @fold_sub_zero_rhs_i32
func.func @fold_sub_zero_rhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %zero = "tosa.const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %sub = "tosa.sub"(%arg0, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %arg0
  return %sub : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_sub_splat_i32
func.func @fold_sub_splat_i32() -> tensor<10xi32> {
  %one = "tosa.const"() {value = dense<1> : tensor<10xi32>} : () -> tensor<10xi32>
  %two = "tosa.const"() {value = dense<2> : tensor<10xi32>} : () -> tensor<10xi32>
  %sub = "tosa.sub"(%one, %two) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
  // CHECK: %[[THREE:.+]] = "tosa.const"() {value = dense<-1> : tensor<10xi32>}
  // CHECK: return %[[THREE]]
  return %sub : tensor<10xi32>
}

// -----

// CHECK-LABEL: @fold_sub_splat_f32
func.func @fold_sub_splat_f32() -> tensor<10xf32> {
  %one = "tosa.const"() {value = dense<1.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %two = "tosa.const"() {value = dense<2.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %sub = "tosa.sub"(%one, %two) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  // CHECK: %[[THREE:.+]] = "tosa.const"() {value = dense<-1.000000e+00> : tensor<10xf32>}
  // CHECK: return %[[THREE]]
  return %sub : tensor<10xf32>
}

// -----

// CHECK-LABEL: @fold_greater_splat_f32
func.func @fold_greater_splat_f32() -> (tensor<10xi1>, tensor<10xi1>) {
  %0 = "tosa.const"() {value = dense<4.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %1 = "tosa.const"() {value = dense<2.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %2 = "tosa.const"() {value = dense<1.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %3 = "tosa.const"() {value = dense<2.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %true = "tosa.greater"(%0, %1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  %false = "tosa.greater"(%2, %3) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  // CHECK-DAG: %[[TRUE:.+]] = "tosa.const"() {value = dense<true> : tensor<10xi1>}
  // CHECK-DAG: %[[FALSE:.+]] = "tosa.const"() {value = dense<false> : tensor<10xi1>}
  // CHECK: return %[[TRUE]], %[[FALSE]]
  return %true, %false : tensor<10xi1>, tensor<10xi1>
}

// -----

// CHECK-LABEL: @fold_greater_splat_i32
func.func @fold_greater_splat_i32() -> (tensor<10xi1>, tensor<10xi1>) {
  %0 = "tosa.const"() {value = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %1 = "tosa.const"() {value = dense<8> : tensor<10xi32>} : () -> tensor<10xi32>
  %2 = "tosa.const"() {value = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %3 = "tosa.const"() {value = dense<-12> : tensor<10xi32>} : () -> tensor<10xi32>
  %false = "tosa.greater"(%0, %1) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  %true = "tosa.greater"(%2, %3) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  // CHECK-DAG: %[[FALSE:.+]] = "tosa.const"() {value = dense<false> : tensor<10xi1>}
  // CHECK-DAG: %[[TRUE:.+]] = "tosa.const"() {value = dense<true> : tensor<10xi1>}
  // CHECK: return %[[FALSE]], %[[TRUE]]
  return %false, %true : tensor<10xi1>, tensor<10xi1>
}

// -----

// CHECK-LABEL: @fold_greater_eq_splat_f32
func.func @fold_greater_eq_splat_f32() -> (tensor<10xi1>, tensor<10xi1>) {
  %0 = "tosa.const"() {value = dense<4.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %1 = "tosa.const"() {value = dense<4.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %2 = "tosa.const"() {value = dense<1.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %3 = "tosa.const"() {value = dense<2.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %true = "tosa.greater_equal"(%0, %1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  %false = "tosa.greater_equal"(%2, %3) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  // CHECK-DAG: %[[TRUE:.+]] = "tosa.const"() {value = dense<true> : tensor<10xi1>}
  // CHECK-DAG: %[[FALSE:.+]] = "tosa.const"() {value = dense<false> : tensor<10xi1>}
  // CHECK: return %[[TRUE]], %[[FALSE]]
  return %true, %false : tensor<10xi1>, tensor<10xi1>
}

// -----

// CHECK-LABEL: @fold_greater_eq_splat_i32
func.func @fold_greater_eq_splat_i32() -> (tensor<10xi1>, tensor<10xi1>) {
  %0 = "tosa.const"() {value = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %1 = "tosa.const"() {value = dense<8> : tensor<10xi32>} : () -> tensor<10xi32>
  %2 = "tosa.const"() {value = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %3 = "tosa.const"() {value = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %true = "tosa.greater_equal"(%2, %3) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  %false = "tosa.greater_equal"(%0, %1) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  // CHECK-DAG: %[[TRUE:.+]] = "tosa.const"() {value = dense<true> : tensor<10xi1>}
  // CHECK-DAG: %[[FALSE:.+]] = "tosa.const"() {value = dense<false> : tensor<10xi1>}
  // CHECK: return %[[TRUE]], %[[FALSE]]
  return %true, %false : tensor<10xi1>, tensor<10xi1>
}

// -----

// CHECK-LABEL: @fold_eq_splat_f32
func.func @fold_eq_splat_f32() -> (tensor<10xi1>, tensor<10xi1>) {
  %0 = "tosa.const"() {value = dense<4.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %1 = "tosa.const"() {value = dense<4.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %2 = "tosa.const"() {value = dense<1.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %3 = "tosa.const"() {value = dense<2.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %true = "tosa.equal"(%0, %1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  %false = "tosa.equal"(%2, %3) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  // CHECK-DAG: %[[TRUE:.+]] = "tosa.const"() {value = dense<true> : tensor<10xi1>}
  // CHECK-DAG: %[[FALSE:.+]] = "tosa.const"() {value = dense<false> : tensor<10xi1>}
  // CHECK: return %[[TRUE]], %[[FALSE]]
  return %true, %false : tensor<10xi1>, tensor<10xi1>
}

// -----

// CHECK-LABEL: @fold_eq_splat_i32
func.func @fold_eq_splat_i32() -> (tensor<10xi1>, tensor<10xi1>) {
  %0 = "tosa.const"() {value = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %1 = "tosa.const"() {value = dense<8> : tensor<10xi32>} : () -> tensor<10xi32>
  %2 = "tosa.const"() {value = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %3 = "tosa.const"() {value = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %true = "tosa.equal"(%2, %3) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  %false = "tosa.equal"(%0, %1) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  // CHECK-DAG: %[[TRUE:.+]] = "tosa.const"() {value = dense<true> : tensor<10xi1>}
  // CHECK-DAG: %[[FALSE:.+]] = "tosa.const"() {value = dense<false> : tensor<10xi1>}
  // CHECK: return %[[TRUE]], %[[FALSE]]
  return %true, %false : tensor<10xi1>, tensor<10xi1>
}

// -----

// CHECK-LABEL: @fold_eq_i32
func.func @fold_eq_i32(%arg0 : tensor<10xi32>) -> (tensor<10xi1>) {
  // CHECK: %[[TRUE:.+]] = "tosa.const"() {value = dense<true> : tensor<10xi1>}
  %0 = "tosa.equal"(%arg0, %arg0) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  // CHECK: return %[[TRUE]]
  return %0 : tensor<10xi1>
}

// -----

func.func @reshape_splat() -> tensor<6x5x4xi32> {
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() {value = dense<42> : tensor<6x5x4xi32>}
  %splat = "tosa.const"() {value = dense<42> : tensor<4x5x6xi32>} : () -> tensor<4x5x6xi32>
  %reshape = "tosa.reshape"(%splat) { new_shape = [6, 5, 4] } : (tensor<4x5x6xi32>) -> tensor<6x5x4xi32>
  // CHECK: return %[[SPLAT]]
  return %reshape : tensor<6x5x4xi32>
}

// -----

// CHECK-LABEL: @slice_splat
func.func @slice_splat() -> tensor<1x1x1xi32> {
  // CHECK: %[[SLICE:.+]] = "tosa.const"() {value = dense<42> : tensor<1x1x1xi32>}
  %splat = "tosa.const"() {value = dense<42> : tensor<4x5x6xi32>} : () -> tensor<4x5x6xi32>
  %slice = "tosa.slice"(%splat) { size = [1, 1, 1], start = [1, 2, 3] } : (tensor<4x5x6xi32>) -> tensor<1x1x1xi32>
  // CHECK: return %[[SLICE]]
  return %slice : tensor<1x1x1xi32>
}

// -----

// CHECK-LABEL: @slice_singleton
func.func @slice_singleton() -> tensor<1x1xi32> {
  %splat = "tosa.const"() {value = dense<[[0, 1, 2], [3, 4, 5], [6, 7 ,8]]> : tensor<3x3xi32>} : () -> tensor<3x3xi32>
  // CHECK: %[[SLICE:.+]] = "tosa.const"() {value = dense<4> : tensor<1x1xi32>}
  %slice = "tosa.slice"(%splat) { size = [1, 1], start = [1, 1] } : (tensor<3x3xi32>) -> tensor<1x1xi32>
  // CHECK: return %[[SLICE]]
  return %slice : tensor<1x1xi32>
}

// -----

// CHECK: func.func @cast_float_to_float
func.func @cast_float_to_float() -> tensor<f16> {
  %splat = "tosa.const"() {value = dense<42.0> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() {value = dense<4.200000e+01> : tensor<f16>} : () -> tensor<f16>
  %cast = "tosa.cast"(%splat) : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[SPLAT]]
  return %cast : tensor<f16>
}

// -----

// CHECK: func.func @cast_int_to_float
func.func @cast_int_to_float() -> tensor<f16> {
  %splat = "tosa.const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() {value = dense<4.000000e+00> : tensor<f16>} : () -> tensor<f16>
  %cast = "tosa.cast"(%splat) : (tensor<i32>) -> tensor<f16>
  // CHECK: return %[[SPLAT]]
  return %cast : tensor<f16>
}

// -----

// CHECK: func.func @cast_float_to_int
func.func @cast_float_to_int() -> tensor<i16> {
  %splat = "tosa.const"() {value = dense<-4.0> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() {value = dense<-4> : tensor<i16>} : () -> tensor<i16>
  %cast = "tosa.cast"(%splat) : (tensor<f32>) -> tensor<i16>
  // CHECK: return %[[SPLAT]]
  return %cast : tensor<i16>
}

// -----

// CHECK: func.func @cast_int_to_int_trunc
func.func @cast_int_to_int_trunc() -> tensor<i16> {
  %splat = "tosa.const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() {value = dense<-1> : tensor<i16>} : () -> tensor<i16>
  %cast = "tosa.cast"(%splat) : (tensor<i32>) -> tensor<i16>
  // CHECK: return %[[SPLAT]]
  return %cast : tensor<i16>
}

// -----

// CHECK: func.func @cast_int_to_int_sign
func.func @cast_int_to_int_sign() -> tensor<i32> {
  %splat = "tosa.const"() {value = dense<-1> : tensor<i16>} : () -> tensor<i16>
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %cast = "tosa.cast"(%splat) : (tensor<i16>) -> tensor<i32>
  // CHECK: return %[[SPLAT]]
  return %cast : tensor<i32>
}

// -----

// CHECK-LABEL: @reverse_splat
func.func @reverse_splat() -> tensor<10xi32> {
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() {value = dense<42> : tensor<10xi32>}
  %splat = "tosa.const"() {value = dense<42> : tensor<10xi32>} : () -> tensor<10xi32>
  %reverse = "tosa.reverse"(%splat) { axis = 0 : i64 } : (tensor<10xi32>) -> tensor<10xi32>
  // CHECK: return %[[SPLAT]]
  return %reverse : tensor<10xi32>
}

// -----

// CHECK-LABEL: @reverse_length_one
func.func @reverse_length_one(%arg0 : tensor<10x1xi32>) -> (tensor<10x1xi32>, tensor<10x1xi32>) {
  %nofold = "tosa.reverse"(%arg0) { axis = 0 : i64 } : (tensor<10x1xi32>) -> tensor<10x1xi32>
  %fold = "tosa.reverse"(%arg0) { axis = 1 : i64 } : (tensor<10x1xi32>) -> tensor<10x1xi32>
  // CHECK: %[[NOFOLD:.+]] = "tosa.reverse"(%arg0) {axis = 0 : i64}
  // CHECK: return %[[NOFOLD]], %arg0
  return %nofold, %fold : tensor<10x1xi32>, tensor<10x1xi32>
}
