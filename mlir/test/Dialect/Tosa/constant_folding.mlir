// RUN: mlir-opt --split-input-file --canonicalize --test-constant-fold %s | FileCheck %s

// -----

// CHECK-LABEL: func @test_const
func.func @test_const(%arg0 : index) -> tensor<4xi32> {
  // CHECK: tosa.const
  %0 = "tosa.const"() {values = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func @test_const_i64
func.func @test_const_i64(%arg0 : index) -> tensor<4xi64> {
  // CHECK: tosa.const
  %0 = "tosa.const"() {values = dense<[3, 0, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  return %0 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func @try_fold_equal_with_unranked_tensor
func.func @try_fold_equal_with_unranked_tensor(%arg0: tensor<4xi32>, %arg1: tensor<1xi32>) -> tensor<*xi1> {
  // CHECK: tosa.equal
  // CHECK-NEXT: return
  %0 = tosa.equal %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_mul_i32
func.func @test_mul_i32() -> tensor<4xi32> {
  // CHECK: %[[VAL:.+]] = "tosa.const"() <{values = dense<[9, 36, 36, 81]> : tensor<4xi32>}>
  // CHECK: return %[[VAL]]
  %lhs = "tosa.const"() {values = dense<[1, 2, -2, -3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %rhs = "tosa.const"() {values = dense<3> : tensor<4xi32>} : () -> tensor<4xi32>
  %shift = "tosa.const"() { values = dense<0> : tensor<1xi8> } : () -> tensor<1xi8>
  %x = tosa.mul %lhs, %rhs, %shift : (tensor<4xi32>, tensor<4xi32>, tensor<1xi8>) -> tensor<4xi32>
  %y = tosa.mul %rhs, %lhs, %shift : (tensor<4xi32>, tensor<4xi32>, tensor<1xi8>) -> tensor<4xi32>
  %result = tosa.mul %x, %y, %shift : (tensor<4xi32>, tensor<4xi32>, tensor<1xi8>) -> tensor<4xi32>

  return %result : tensor<4xi32>
}

// -----

// CHECK-LABEL: test_mul_i32_shift
func.func @test_mul_i32_shift() -> tensor<4xi32> {
  // CHECK: %[[VAL:.+]] = "tosa.const"() <{values = dense<[2550, 8100, 2, 2025]> : tensor<4xi32>}>
  // CHECK: return %[[VAL]]
  %lhs = "tosa.const"() {values = dense<[135, 240, -4, -120]> : tensor<4xi32>} : () -> tensor<4xi32>
  %rhs = "tosa.const"() {values = dense<3> : tensor<4xi32>} : () -> tensor<4xi32>
  %shift = "tosa.const"() { values = dense<2> : tensor<1xi8> } : () -> tensor<1xi8>
  %x = tosa.mul %lhs, %rhs, %shift : (tensor<4xi32>, tensor<4xi32>, tensor<1xi8>) -> tensor<4xi32>
  %y = tosa.mul %rhs, %lhs, %shift : (tensor<4xi32>, tensor<4xi32>, tensor<1xi8>) -> tensor<4xi32>
  %result = tosa.mul %x, %y, %shift : (tensor<4xi32>, tensor<4xi32>, tensor<1xi8>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// -----

// CHECK-LABEL: test_mul_f32
func.func @test_mul_f32() -> tensor<4xf32> {
  // CHECK: %[[VAL:.+]] = "tosa.const"() <{values = dense<[2.304000e+01, 58.9824028, 1.6384002, 14.7456007]> : tensor<4xf32>}>
  // CHECK: return %[[VAL]]
  %lhs = "tosa.const"() {values = dense<[1.5, 2.4, -0.4, -1.2]> : tensor<4xf32>} : () -> tensor<4xf32>
  %rhs = "tosa.const"() {values = dense<3.2> : tensor<4xf32>} : () -> tensor<4xf32>
  %shift = "tosa.const"() { values = dense<0> : tensor<1xi8> } : () -> tensor<1xi8>
  %x = tosa.mul %lhs, %rhs, %shift : (tensor<4xf32>, tensor<4xf32>, tensor<1xi8>) -> tensor<4xf32>
  %y = tosa.mul %rhs, %lhs, %shift : (tensor<4xf32>, tensor<4xf32>, tensor<1xi8>) -> tensor<4xf32>
  %result = tosa.mul %x, %y, %shift : (tensor<4xf32>, tensor<4xf32>, tensor<1xi8>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// -----

// CHECK-LABEL: test_add_f32
func.func @test_add_f32() -> tensor<4xf32> {
  // CHECK: %[[VAL:.+]] = "tosa.const"() <{values = dense<[7.500000e+00, 9.300000e+00, 3.69999981, 2.100000e+00]> : tensor<4xf32>}>
  // CHECK: return %[[VAL]]
  %cst = "tosa.const"() {values = dense<[1.5, 2.4, -0.4, -1.2]> : tensor<4xf32>} : () -> tensor<4xf32>
  %splat1 = "tosa.const"() {values = dense<3.2> : tensor<4xf32>} : () -> tensor<4xf32>
  %splat2 = "tosa.const"() {values = dense<1.3> : tensor<4xf32>} : () -> tensor<4xf32>
  %x = tosa.add %cst, %splat1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %y = tosa.add %splat2, %cst : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %result = tosa.add %x, %y : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// -----

// CHECK-LABEL: test_add_i32
func.func @test_add_i32() -> tensor<4xi32> {
  // CHECK: %[[VAL:.+]] = "tosa.const"() <{values = dense<[75, 93, 37, 21]> : tensor<4xi32>}>
  // CHECK: return %[[VAL]]
  %cst = "tosa.const"() {values = dense<[15, 24, -4, -12]> : tensor<4xi32>} : () -> tensor<4xi32>
  %splat1 = "tosa.const"() {values = dense<32> : tensor<4xi32>} : () -> tensor<4xi32>
  %splat2 = "tosa.const"() {values = dense<13> : tensor<4xi32>} : () -> tensor<4xi32>
  %x = tosa.add %cst, %splat1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %y = tosa.add %splat2, %cst : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %result = tosa.add %x, %y : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// -----

// CHECK-LABEL: test_sub_f32
func.func @test_sub_f32() -> tensor<4xf32> {
  // CHECK: %[[VAL:.+]] = "tosa.const"() <{values = dense<[-1.500000e+00, 0.300000191, -5.300000e+00, -6.900000e+00]> : tensor<4xf32>}>
  // CHECK: return %[[VAL]]
  %cst = "tosa.const"() {values = dense<[1.5, 2.4, -0.4, -1.2]> : tensor<4xf32>} : () -> tensor<4xf32>
  %splat1 = "tosa.const"() {values = dense<3.2> : tensor<4xf32>} : () -> tensor<4xf32>
  %splat2 = "tosa.const"() {values = dense<1.3> : tensor<4xf32>} : () -> tensor<4xf32>
  %x = tosa.sub %cst, %splat1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %y = tosa.sub %splat2, %cst : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %result = tosa.sub %x, %y : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// -----

// CHECK-LABEL: test_sub_i32
func.func @test_sub_i32() -> tensor<4xi32> {
  // CHECK: %[[VAL:.+]] = "tosa.const"() <{values = dense<[-15, 3, -53, -69]> : tensor<4xi32>}>
  // CHECK: return %[[VAL]]
  %cst = "tosa.const"() {values = dense<[15, 24, -4, -12]> : tensor<4xi32>} : () -> tensor<4xi32>
  %splat1 = "tosa.const"() {values = dense<32> : tensor<4xi32>} : () -> tensor<4xi32>
  %splat2 = "tosa.const"() {values = dense<13> : tensor<4xi32>} : () -> tensor<4xi32>
  %x = tosa.sub %cst, %splat1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %y = tosa.sub %splat2, %cst : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %result = tosa.sub %x, %y : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// -----

// CHECK-LABEL: test_greater_f32
func.func @test_greater_f32() -> (tensor<4xi1>, tensor<4xi1>, tensor<4xi1>) {
  // CHECK: %[[VAL_0:.+]] = "tosa.const"() <{values = dense<[false, true, false, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: %[[VAL_1:.+]] = "tosa.const"() <{values = dense<[false, false, true, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: %[[VAL_2:.+]] = "tosa.const"() <{values = dense<[false, true, true, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: return %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]
  %cst1 = "tosa.const"() {values = dense<[1.5, 2.4, -0.4, -1.2]> : tensor<4xf32>} : () -> tensor<4xf32>
  %splat = "tosa.const"() {values = dense<1.5> : tensor<4xf32>} : () -> tensor<4xf32>
  %cst2 = "tosa.const"() {values = dense<[1.7, 2.3, -0.5, -1.1]> : tensor<4xf32>} : () -> tensor<4xf32>
  %x = tosa.greater %cst1, %splat : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %y = tosa.greater %splat, %cst1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %z = tosa.greater %cst1, %cst2 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %x, %y, %z : tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
}

// -----

// CHECK-LABEL: test_greater_i32
func.func @test_greater_i32() -> (tensor<4xi1>, tensor<4xi1>, tensor<4xi1>) {
  // CHECK: %[[VAL_0:.+]] = "tosa.const"() <{values = dense<[false, true, false, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: %[[VAL_1:.+]] = "tosa.const"() <{values = dense<[false, false, true, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: %[[VAL_2:.+]] = "tosa.const"() <{values = dense<[false, true, true, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: return %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]
  %cst1 = "tosa.const"() {values = dense<[15, 24, -4, -12]> : tensor<4xi32>} : () -> tensor<4xi32>
  %cst2 = "tosa.const"() {values = dense<[17, 23, -5, -11]> : tensor<4xi32>} : () -> tensor<4xi32>
  %splat = "tosa.const"() {values = dense<15> : tensor<4xi32>} : () -> tensor<4xi32>
  %x = tosa.greater %cst1, %splat : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %y = tosa.greater %splat, %cst1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %z = tosa.greater %cst1, %cst2 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %x, %y, %z : tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
}

// -----

// CHECK-LABEL: test_greater_equal_f32
func.func @test_greater_equal_f32() -> (tensor<4xi1>, tensor<4xi1>, tensor<4xi1>) {
  // CHECK: %[[VAL_0:.+]] = "tosa.const"() <{values = dense<[true, true, false, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: %[[VAL_1:.+]] = "tosa.const"() <{values = dense<[true, false, true, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: %[[VAL_2:.+]] = "tosa.const"() <{values = dense<[true, true, true, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: return %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]
  %cst1 = "tosa.const"() {values = dense<[1.5, 2.4, -0.4, -1.2]> : tensor<4xf32>} : () -> tensor<4xf32>
  %splat = "tosa.const"() {values = dense<1.5> : tensor<4xf32>} : () -> tensor<4xf32>
  %cst2 = "tosa.const"() {values = dense<[1.4, 2.4, -0.5, -1.1]> : tensor<4xf32>} : () -> tensor<4xf32>
  %x = tosa.greater_equal %cst1, %splat : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %y = tosa.greater_equal %splat, %cst1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %z = tosa.greater_equal %cst1, %cst2 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %x, %y, %z : tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
}

// -----

// CHECK-LABEL: test_greater_equal_i32
func.func @test_greater_equal_i32() -> (tensor<4xi1>, tensor<4xi1>, tensor<4xi1>) {
  // CHECK: %[[VAL_0:.+]] = "tosa.const"() <{values = dense<[false, true, false, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: %[[VAL_1:.+]] = "tosa.const"() <{values = dense<[true, false, true, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: %[[VAL_2:.+]] = "tosa.const"() <{values = dense<[true, true, true, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: return %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]
  %cst1 = "tosa.const"() {values = dense<[15, 24, -4, -12]> : tensor<4xi32>} : () -> tensor<4xi32>
  %splat = "tosa.const"() {values = dense<16> : tensor<4xi32>} : () -> tensor<4xi32>
  %cst2 = "tosa.const"() {values = dense<[14, 24, -5, -11]> : tensor<4xi32>} : () -> tensor<4xi32>
  %x = tosa.greater_equal %cst1, %splat : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %y = tosa.greater_equal %splat, %cst1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %z = tosa.greater_equal %cst1, %cst2 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %x, %y, %z : tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
}

// -----

// CHECK-LABEL: test_equal_f32
func.func @test_equal_f32() -> (tensor<4xi1>, tensor<4xi1>, tensor<4xi1>) {
  // CHECK: %[[VAL_0:.+]] = "tosa.const"() <{values = dense<[true, false, false, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: %[[VAL_1:.+]] = "tosa.const"() <{values = dense<[false, true, false, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: return %[[VAL_0]], %[[VAL_0]], %[[VAL_1]]
  %cst1 = "tosa.const"() {values = dense<[1.5, 2.4, -0.4, -1.2]> : tensor<4xf32>} : () -> tensor<4xf32>
  %splat = "tosa.const"() {values = dense<1.5> : tensor<4xf32>} : () -> tensor<4xf32>
  %cst2 = "tosa.const"() {values = dense<[1.4, 2.4, -0.5, -1.2]> : tensor<4xf32>} : () -> tensor<4xf32>
  %x = tosa.equal %cst1, %splat : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %y = tosa.equal %splat, %cst1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %z = tosa.equal %cst1, %cst2 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %x, %y, %z : tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
}

// -----

// CHECK-LABEL: test_equal_i32
func.func @test_equal_i32() -> (tensor<4xi1>, tensor<4xi1>, tensor<4xi1>) {
  // CHECK: %[[VAL_0:.+]] = "tosa.const"() <{values = dense<[true, false, false, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: %[[VAL_1:.+]] = "tosa.const"() <{values = dense<[false, true, false, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
  // CHECK: return %[[VAL_0]], %[[VAL_0]], %[[VAL_1]]
  %cst1 = "tosa.const"() {values = dense<[15, 24, -4, -12]> : tensor<4xi32>} : () -> tensor<4xi32>
  %splat = "tosa.const"() {values = dense<15> : tensor<4xi32>} : () -> tensor<4xi32>
  %cst2 = "tosa.const"() {values = dense<[14, 24, -5, -12]> : tensor<4xi32>} : () -> tensor<4xi32>
  %x = tosa.equal %cst1, %splat : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %y = tosa.equal %splat, %cst1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %z = tosa.equal %cst1, %cst2 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %x, %y, %z : tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
}
