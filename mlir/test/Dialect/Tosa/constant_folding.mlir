// RUN: mlir-opt --split-input-file --test-single-fold %s | FileCheck %s

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
func.func @try_fold_equal_with_unranked_tensor(%arg0: tensor<4xi32>, %arg1: tensor<1xi32>) {
  // CHECK: tosa.equal
  // CHECK-NEXT: return
  %0 = tosa.equal %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi1>
  return
}

// -----

// CHECK-LABEL: @fold_add_zero_rhs_f32
func.func @fold_add_zero_rhs_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %zero = "tosa.const"() {values = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %add = tosa.add %arg0, %zero : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %arg0
  return %add : tensor<f32>
}

// -----

// CHECK-LABEL: @fold_add_zero_lhs_f32
func.func @fold_add_zero_lhs_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %zero = "tosa.const"() {values = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %add = tosa.add %zero, %arg0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %arg0
  return %add : tensor<f32>
}

// -----

// CHECK-LABEL: @fold_add_zero_rhs_i32
func.func @fold_add_zero_rhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<i32>} : () -> tensor<i32>
  %add = tosa.add %arg0, %zero : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %arg0
  return %add : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_add_zero_lhs_i32
func.func @fold_add_zero_lhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<i32>} : () -> tensor<i32>
  %add = tosa.add %zero, %arg0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %arg0
  return %add : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_add_splat_i32
func.func @fold_add_splat_i32() -> tensor<10xi32> {
  %one = "tosa.const"() {values = dense<1> : tensor<10xi32>} : () -> tensor<10xi32>
  %two = "tosa.const"() {values = dense<2> : tensor<10xi32>} : () -> tensor<10xi32>
  %add = tosa.add %one, %two : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
  // CHECK: %[[THREE:.+]] = "tosa.const"() <{values = dense<3> : tensor<10xi32>}
  // CHECK: return %[[THREE]]
  return %add : tensor<10xi32>
}

// -----

// CHECK-LABEL: @fold_add_splat_f32
func.func @fold_add_splat_f32() -> tensor<10xf32> {
  %one = "tosa.const"() {values = dense<1.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %two = "tosa.const"() {values = dense<2.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %add = tosa.add %one, %two : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  // CHECK: %[[THREE:.+]] = "tosa.const"() <{values = dense<3.000000e+00>
  // CHECK: return %[[THREE]]
  return %add : tensor<10xf32>
}

// -----

// CHECK-LABEL: @fold_add_splat_i32_positive_overflow
func.func @fold_add_splat_i32_positive_overflow() -> tensor<10xi32> {
  %one = "tosa.const"() {values = dense<2147483647> : tensor<10xi32>} : () -> tensor<10xi32>
  %two = "tosa.const"() {values = dense<1> : tensor<10xi32>} : () -> tensor<10xi32>
  // CHECK: tosa.add
  %add = tosa.add %one, %two : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
  return %add : tensor<10xi32>
}

// -----

// CHECK-LABEL: @fold_add_splat_i32_negative_overflow
func.func @fold_add_splat_i32_negative_overflow() -> tensor<10xi32> {
  %one = "tosa.const"() {values = dense<-1> : tensor<10xi32>} : () -> tensor<10xi32>
  %two = "tosa.const"() {values = dense<-2147483648> : tensor<10xi32>} : () -> tensor<10xi32>
  // CHECK: tosa.add
  %add = tosa.add %one, %two : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
  return %add : tensor<10xi32>
}

// -----

// CHECK-LABEL: @fold_add_splat_ui8
func.func @fold_add_splat_ui8() -> tensor<10xui8> {
  %one = "tosa.const"() {values = dense<1> : tensor<10xui8>} : () -> tensor<10xui8>
  %two = "tosa.const"() {values = dense<254> : tensor<10xui8>} : () -> tensor<10xui8>
  // CHECK: "tosa.const"() <{values = dense<255> : tensor<10xui8>}> : () -> tensor<10xui8>
  %add = tosa.add %one, %two : (tensor<10xui8>, tensor<10xui8>) -> tensor<10xui8>
  return %add : tensor<10xui8>
}

// -----

// CHECK-LABEL: @fold_add_splat_ui8_overflow
func.func @fold_add_splat_ui8_overflow() -> tensor<10xui8> {
  %one = "tosa.const"() {values = dense<2> : tensor<10xui8>} : () -> tensor<10xui8>
  %two = "tosa.const"() {values = dense<254> : tensor<10xui8>} : () -> tensor<10xui8>
  // CHECK: tosa.add
  %add = tosa.add %one, %two : (tensor<10xui8>, tensor<10xui8>) -> tensor<10xui8>
  return %add : tensor<10xui8>
}

// -----

// CHECK-LABEL: @no_fold_add_unknown_broadcast_zero_lhs
func.func @no_fold_add_unknown_broadcast_zero_lhs(%arg0: tensor<1x4xi32>) -> tensor<2x4xi32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<2x1xi32>} : () -> tensor<2x1xi32>
  %div = tosa.add %zero, %arg0 : (tensor<2x1xi32>, tensor<1x4xi32>) -> tensor<2x4xi32>
  // CHECK: tosa.add
  return %div : tensor<2x4xi32>
}

// -----

// CHECK-LABEL: @no_fold_dynamic_add_unknown_broadcast_zero_lhs
func.func @no_fold_dynamic_add_unknown_broadcast_zero_lhs(%arg0: tensor<?x4xi32>) -> tensor<?x4xi32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<2x1xi32>} : () -> tensor<2x1xi32>
  %div = tosa.add %zero, %arg0 : (tensor<2x1xi32>, tensor<?x4xi32>) -> tensor<?x4xi32>
  // CHECK: tosa.add
  return %div : tensor<?x4xi32>
}

// -----

// CHECK-LABEL: @fold_dynamic_add_broadcast_zero_lhs
func.func @fold_dynamic_add_broadcast_zero_lhs(%arg0: tensor<?x17xi32>) -> tensor<?x17xi32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %div = tosa.add %zero, %arg0 : (tensor<1x1xi32>, tensor<?x17xi32>) -> tensor<?x17xi32>
  // CHECK: return %arg0
  return %div : tensor<?x17xi32>
}

// -----

// CHECK-LABEL: @no_fold_dynamic_add_unknown_broadcast_zero_rhs
func.func @no_fold_dynamic_add_unknown_broadcast_zero_rhs(%arg0: tensor<?x4xi32>) -> tensor<?x4xi32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<2x1xi32>} : () -> tensor<2x1xi32>
  %div = tosa.add %arg0, %zero : (tensor<?x4xi32>, tensor<2x1xi32>) -> tensor<?x4xi32>
  // CHECK: tosa.add
  return %div : tensor<?x4xi32>
}

// -----

// CHECK-LABEL: @fold_dynamic_add_broadcast_zero_rhs
func.func @fold_dynamic_add_broadcast_zero_rhs(%arg0: tensor<?x17xi32>) -> tensor<?x17xi32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<1x17xi32>} : () -> tensor<1x17xi32>
  %div = tosa.add %arg0, %zero : (tensor<?x17xi32>, tensor<1x17xi32>) -> tensor<?x17xi32>
  // CHECK: return %arg0
  return %div : tensor<?x17xi32>
}

// -----

// CHECK-LABEL: @fold_div_zero_lhs_i32
func.func @fold_div_zero_lhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[ZERO:.+]] = "tosa.const"() <{values = dense<0>
  %div = tosa.intdiv %zero, %arg0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %[[ZERO]]
  return %div : tensor<i32>
}

// -----

// CHECK-LABEL: @no_fold_dynamic_div_zero_lhs
func.func @no_fold_dynamic_div_zero_lhs(%arg0: tensor<?x4xi32>) -> tensor<?x4xi32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  // CHECK: tosa.intdiv
  %div = tosa.intdiv %zero, %arg0 : (tensor<1x1xi32>, tensor<?x4xi32>) -> tensor<?x4xi32>
  return %div : tensor<?x4xi32>
}

// -----

// CHECK-LABEL: @fold_div_zero_lhs_broadcast
func.func @fold_div_zero_lhs_broadcast(%arg0: tensor<2x4xi32>) -> tensor<2x4xi32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  // CHECK: %[[ZERO:.+]] = "tosa.const"() <{values = dense<0> : tensor<2x4xi32>
  %div = tosa.intdiv %zero, %arg0 : (tensor<1x1xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
  // CHECK: return %[[ZERO]]
  return %div : tensor<2x4xi32>
}

// -----

// CHECK-LABEL: @no_fold_div_unknown_broadcast_one_rhs
func.func @no_fold_div_unknown_broadcast_one_rhs(%arg0: tensor<1x4xi32>) -> tensor<2x4xi32> {
  %one = "tosa.const"() {values = dense<1> : tensor<2x1xi32>} : () -> tensor<2x1xi32>
  // CHECK: tosa.intdiv
  %div = tosa.intdiv %arg0, %one : (tensor<1x4xi32>, tensor<2x1xi32>) -> tensor<2x4xi32>
  return %div : tensor<2x4xi32>
}

// -----

// CHECK-LABEL: @fold_div_broadcast_one_rhs
func.func @fold_div_broadcast_one_rhs(%arg0: tensor<?x17xi32>) -> tensor<?x17xi32> {
  %one = "tosa.const"() {values = dense<1> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %div = tosa.intdiv %arg0, %one : (tensor<?x17xi32>, tensor<1x1xi32>) -> tensor<?x17xi32>
  // CHECK: return %arg0
  return %div : tensor<?x17xi32>
}

// -----

// CHECK-LABEL: @fold_div_one_rhs_i32
func.func @fold_div_one_rhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %one = "tosa.const"() {values = dense<1> : tensor<i32>} : () -> tensor<i32>
  %div = tosa.intdiv %arg0, %one : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %arg0
  return %div : tensor<i32>
}

// -----

// CHECK-LABEL: @no_fold_dynamic_div_unknown_broadcast_one_rhs
func.func @no_fold_dynamic_div_unknown_broadcast_one_rhs(%arg0: tensor<?x4xi32>) -> tensor<?x4xi32> {
  %one = "tosa.const"() {values = dense<1> : tensor<2x1xi32>} : () -> tensor<2x1xi32>
  // CHECK: tosa.intdiv
  %div = tosa.intdiv %arg0, %one : (tensor<?x4xi32>, tensor<2x1xi32>) -> tensor<?x4xi32>
  return %div : tensor<?x4xi32>
}

// -----

// CHECK-LABEL: @fold_div_splat_i32
func.func @fold_div_splat_i32() -> tensor<i32> {
  %lhs = "tosa.const"() {values = dense<10> : tensor<i32>} : () -> tensor<i32>
  %rhs = "tosa.const"() {values = dense<-3> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() <{values = dense<-3>
  %div = tosa.intdiv %lhs, %rhs : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %[[SPLAT]]
  return %div : tensor<i32>
}

// -----


// CHECK-LABEL: @fold_mul_zero_rhs_f32
func.func @fold_mul_zero_rhs_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %zero = "tosa.const"() {values = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[ZERO:.+]] = "tosa.const"() <{values = dense<0.000000e+00>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %mul = tosa.mul %arg0, %zero, %shift : (tensor<f32>, tensor<f32>, tensor<1xi8>) -> tensor<f32>
  // CHECK: return %[[ZERO]]
  return %mul : tensor<f32>
}

// -----

// CHECK-LABEL: @fold_mul_zero_lhs_f32
func.func @fold_mul_zero_lhs_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %zero = "tosa.const"() {values = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[ZERO:.+]] = "tosa.const"() <{values = dense<0.000000e+00>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %mul = tosa.mul %zero, %arg0, %shift : (tensor<f32>, tensor<f32>, tensor<1xi8>) -> tensor<f32>
  // CHECK: return %[[ZERO]]
  return %mul : tensor<f32>
}

// -----

// CHECK-LABEL: @fold_mul_zero_rhs_i32
func.func @fold_mul_zero_rhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<i32>} : () -> tensor<i32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: %[[ZERO:.+]] = "tosa.const"() <{values = dense<0>
  %mul = tosa.mul %arg0, %zero, %shift : (tensor<i32>, tensor<i32>, tensor<1xi8>) -> tensor<i32>
  // CHECK: return %[[ZERO]]
  return %mul : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_mul_zero_lhs_i32
func.func @fold_mul_zero_lhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<i32>} : () -> tensor<i32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: %[[ZERO:.+]] = "tosa.const"() <{values = dense<0>
  %mul = tosa.mul %zero, %arg0, %shift : (tensor<i32>, tensor<i32>, tensor<1xi8>) -> tensor<i32>
  // CHECK: return %[[ZERO]]
  return %mul : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_mul_one_rhs_f32
func.func @fold_mul_one_rhs_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %one = "tosa.const"() {values = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %mul = tosa.mul %arg0, %one, %shift : (tensor<f32>, tensor<f32>, tensor<1xi8>) -> tensor<f32>
  // CHECK: return %arg0
  return %mul : tensor<f32>
}

// -----

// CHECK-LABEL: @no_fold_mul_unknown_broadcast_one_rhs_f32
func.func @no_fold_mul_unknown_broadcast_one_rhs_f32(%arg0: tensor<1x4xf32>) -> tensor<2x4xf32> {
  %one = "tosa.const"() {values = dense<1.0> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: tosa.mul
  %mul = tosa.mul %arg0, %one, %shift : (tensor<1x4xf32>, tensor<2x1xf32>, tensor<1xi8>) -> tensor<2x4xf32>
  return %mul : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: @no_fold_dynamic_mul_unknown_broadcast_one_rhs_f32
func.func @no_fold_dynamic_mul_unknown_broadcast_one_rhs_f32(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %one = "tosa.const"() {values = dense<1.0> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: tosa.mul
  %mul = tosa.mul %arg0, %one, %shift : (tensor<?x4xf32>, tensor<2x1xf32>, tensor<1xi8>) -> tensor<?x4xf32>
  return %mul : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @fold_mul_one_lhs_f32
func.func @fold_mul_one_lhs_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %one = "tosa.const"() {values = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %mul = tosa.mul %one, %arg0, %shift : (tensor<f32>, tensor<f32>, tensor<1xi8>) -> tensor<f32>
  // CHECK: return %arg0
  return %mul : tensor<f32>
}

// -----

// CHECK-LABEL: @fold_mul_one_rhs_i32
func.func @fold_mul_one_rhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %one = "tosa.const"() {values = dense<64> : tensor<i32>} : () -> tensor<i32>
  %shift = "tosa.const"() {values = dense<6> : tensor<1xi8>} : () -> tensor<1xi8>
  %mul = tosa.mul %arg0, %one, %shift : (tensor<i32>, tensor<i32>, tensor<1xi8>) -> tensor<i32>
  // CHECK: return %arg0
  return %mul : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_mul_one_lhs_i32
func.func @fold_mul_one_lhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %one = "tosa.const"() {values = dense<64> : tensor<i32>} : () -> tensor<i32>
  %shift = "tosa.const"() {values = dense<6> : tensor<1xi8>} : () -> tensor<1xi8>
  %mul = tosa.mul %one, %arg0, %shift : (tensor<i32>, tensor<i32>, tensor<1xi8>) -> tensor<i32>
  // CHECK: return %arg0
  return %mul : tensor<i32>
}

// -----

// CHECK-LABEL: @no_fold_mul_unknown_broadcast_one_lhs_f32
func.func @no_fold_mul_unknown_broadcast_one_lhs_f32(%arg0: tensor<1x4xf32>) -> tensor<2x4xf32> {
  %one = "tosa.const"() {values = dense<1.0> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: tosa.mul
  %mul = tosa.mul %one, %arg0, %shift : (tensor<2x1xf32>, tensor<1x4xf32>, tensor<1xi8>) -> tensor<2x4xf32>
  return %mul : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: @no_fold_dynamic_mul_unknown_broadcast_one_lhs_f32
func.func @no_fold_dynamic_mul_unknown_broadcast_one_lhs_f32(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %one = "tosa.const"() {values = dense<1.0> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: tosa.mul
  %mul = tosa.mul %one, %arg0, %shift : (tensor<2x1xf32>, tensor<?x4xf32>, tensor<1xi8>) -> tensor<?x4xf32>
  return %mul : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @fold_mul_one_float
func.func @fold_mul_one_float(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.mul
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %ones = "tosa.const"() {values = dense<1.0> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %1 = tosa.mul %arg0, %ones, %shift : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<1xi8>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @fold_mul_bcast_one_float
func.func @fold_mul_bcast_one_float(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.mul
  %ones = "tosa.const"() {values = dense<1.0> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %1 = tosa.mul %ones, %arg0, %shift : (tensor<1x1xf32>, tensor<2x3xf32>, tensor<1xi8>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @fold_mul_one_int
func.func @fold_mul_one_int(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.mul
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %ones = "tosa.const"() {values = dense<1> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %1 = tosa.mul %arg0, %ones, %shift : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<1xi8>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @fold_mul_one_int_and_shift
func.func @fold_mul_one_int_and_shift(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<1> : tensor<2x3xi32>}>
  // CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<31> : tensor<1xi8>}>
  // CHECK: %[[VAL_3:.*]] = tosa.mul %arg0, %[[VAL_1]], %[[VAL_2]] : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<1xi8>)
  %ones = "tosa.const"() {values = dense<1> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %shift = "tosa.const"() <{values = dense<31> : tensor<1xi8>}> : () -> tensor<1xi8>
  %1 = tosa.mul %arg0, %ones, %shift : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<1xi8>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @fold_mul_zero_broadcast
func.func @fold_mul_zero_broadcast(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
  // CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<2x3xf32>}
  // CHECK-NOT: tosa.mul
  %zeros = "tosa.const"() {values = dense<0.0> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %1 = tosa.mul %arg0, %zeros, %shift : (tensor<2x3xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<2x3xf32>

  // CHECK-NOT: tosa.mul
  // CHECK: return %[[ZERO]], %[[ZERO]]
  %2 = tosa.mul %zeros, %arg0, %shift : (tensor<1x1xf32>, tensor<2x3xf32>, tensor<1xi8>) -> tensor<2x3xf32>
  return %1, %2 : tensor<2x3xf32>, tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @fold_mul_zero_dynamic_nofold
// CHECK-SAME:                    %[[ARG0:.*]]: tensor<?x17xf32>) -> tensor<?x17xf32> {
// CHECK:           %[[ZERO:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK:           %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[MUL:.*]] = tosa.mul %[[ARG0]], %[[ZERO]], %[[SHIFT]] : (tensor<?x17xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<?x17xf32>
// CHECK:           return %[[MUL]]
func.func @fold_mul_zero_dynamic_nofold(%arg0: tensor<?x17xf32>) -> tensor<?x17xf32> {
  %0 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
  %1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %2 = tosa.mul %arg0, %0, %1 : (tensor<?x17xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<?x17xf32>
  return %2 : tensor<?x17xf32>
}

// -----

// CHECK-LABEL: @fold_mul_one_dynamic_fold
// CHECK-SAME:                    %[[ARG0:.*]]: tensor<?x17xf32>) -> tensor<?x17xf32> {
// CHECK:           return %[[ARG0]]
func.func @fold_mul_one_dynamic_fold(%arg0: tensor<?x17xf32>) -> tensor<?x17xf32> {
  %0 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
  %1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %2 = tosa.mul %arg0, %0, %1 : (tensor<?x17xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<?x17xf32>
  return %2 : tensor<?x17xf32>
}

// -----

// CHECK-LABEL: @fold_mul_splat_i8
func.func @fold_mul_splat_i8() -> tensor<10xi32> {
  %one = "tosa.const"() {values = dense<17> : tensor<10xi8>} : () -> tensor<10xi8>
  %two = "tosa.const"() {values = dense<32> : tensor<10xi8>} : () -> tensor<10xi8>
  %shift = "tosa.const"() {values = dense<3> : tensor<1xi8>} : () -> tensor<1xi8>
  %mul = tosa.mul %one, %two, %shift : (tensor<10xi8>, tensor<10xi8>, tensor<1xi8>) -> tensor<10xi32>
  // CHECK: %[[SIXTY_EIGHT:.+]] = "tosa.const"() <{values = dense<68> : tensor<10xi32>}
  // CHECK: return %[[SIXTY_EIGHT]]
  return %mul : tensor<10xi32>
}

// -----

// CHECK-LABEL: @fold_mul_splat_f32
func.func @fold_mul_splat_f32() -> tensor<10xf32> {
  %one = "tosa.const"() {values = dense<3.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %two = "tosa.const"() {values = dense<2.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %mul = tosa.mul %one, %two, %shift : (tensor<10xf32>, tensor<10xf32>, tensor<1xi8>) -> tensor<10xf32>
  // CHECK: %[[SIX:.+]] = "tosa.const"() <{values = dense<6.000000e+00> : tensor<10xf32>}
  // CHECK: return %[[SIX]]
  return %mul : tensor<10xf32>
}

// -----

// CHECK-LABEL: @fold_sub_zero_rhs_f32
func.func @fold_sub_zero_rhs_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %zero = "tosa.const"() {values = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %sub = tosa.sub %arg0, %zero : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %arg0
  return %sub : tensor<f32>
}

// -----

// CHECK-LABEL: @fold_sub_zero_rhs_i32
func.func @fold_sub_zero_rhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<i32>} : () -> tensor<i32>
  %sub = tosa.sub %arg0, %zero : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %arg0
  return %sub : tensor<i32>
}

// -----

// CHECK-LABEL: @no_fold_sub_unknown_broadcast_zero_rhs
func.func @no_fold_sub_unknown_broadcast_zero_rhs(%arg0: tensor<1x4xi32>) -> tensor<2x4xi32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<2x1xi32>} : () -> tensor<2x1xi32>
  %div = tosa.sub %arg0, %zero : (tensor<1x4xi32>, tensor<2x1xi32>) -> tensor<2x4xi32>
  // CHECK: tosa.sub
  return %div : tensor<2x4xi32>
}

// -----

// CHECK-LABEL: @no_fold_dynamic_sub_unknown_broadcast_zero_rhs
func.func @no_fold_dynamic_sub_unknown_broadcast_zero_rhs(%arg0: tensor<?x4xi32>) -> tensor<?x4xi32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<2x1xi32>} : () -> tensor<2x1xi32>
  %div = tosa.sub %arg0, %zero : (tensor<?x4xi32>, tensor<2x1xi32>) -> tensor<?x4xi32>
  // CHECK: tosa.sub
  return %div : tensor<?x4xi32>
}

// -----

// CHECK-LABEL: @fold_dynamic_sub_broadcast_zero_rhs
func.func @fold_dynamic_sub_broadcast_zero_rhs(%arg0: tensor<?x17xi32>) -> tensor<?x17xi32> {
  %zero = "tosa.const"() {values = dense<0> : tensor<1x17xi32>} : () -> tensor<1x17xi32>
  %div = tosa.sub %arg0, %zero : (tensor<?x17xi32>, tensor<1x17xi32>) -> tensor<?x17xi32>
  // CHECK: return %arg0
  return %div : tensor<?x17xi32>
}

// -----

// CHECK-LABEL: @fold_sub_splat_i32
func.func @fold_sub_splat_i32() -> tensor<10xi32> {
  %one = "tosa.const"() {values = dense<1> : tensor<10xi32>} : () -> tensor<10xi32>
  %two = "tosa.const"() {values = dense<2> : tensor<10xi32>} : () -> tensor<10xi32>
  %sub = tosa.sub %one, %two : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
  // CHECK: %[[NEGATIVE_ONE:.+]] = "tosa.const"() <{values = dense<-1> : tensor<10xi32>}
  // CHECK: return %[[NEGATIVE_ONE]]
  return %sub : tensor<10xi32>
}

// -----

// CHECK-LABEL: @fold_sub_splat_f32
func.func @fold_sub_splat_f32() -> tensor<10xf32> {
  %one = "tosa.const"() {values = dense<1.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %two = "tosa.const"() {values = dense<2.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %sub = tosa.sub %one, %two : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  // CHECK: %[[NEGATIVE_ONE:.+]] = "tosa.const"() <{values = dense<-1.000000e+00> : tensor<10xf32>}
  // CHECK: return %[[NEGATIVE_ONE]]
  return %sub : tensor<10xf32>
}

// -----

// CHECK-LABEL: @fold_sub_splat_i32_positive_overflow
func.func @fold_sub_splat_i32_positive_overflow() -> tensor<10xi32> {
  %one = "tosa.const"() {values = dense<2147483647> : tensor<10xi32>} : () -> tensor<10xi32>
  %two = "tosa.const"() {values = dense<-1> : tensor<10xi32>} : () -> tensor<10xi32>
  // CHECK: tosa.sub
  %sub = tosa.sub %one, %two : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
  return %sub : tensor<10xi32>
}

// -----

// CHECK-LABEL: @fold_sub_splat_i32_negative_overflow
func.func @fold_sub_splat_i32_negative_overflow() -> tensor<10xi32> {
  %one = "tosa.const"() {values = dense<-2147483648> : tensor<10xi32>} : () -> tensor<10xi32>
  %two = "tosa.const"() {values = dense<1> : tensor<10xi32>} : () -> tensor<10xi32>
  // CHECK: tosa.sub
  %sub = tosa.sub %one, %two : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
  return %sub : tensor<10xi32>
}

// -----

// CHECK-LABEL: @fold_sub_splat_ui8
func.func @fold_sub_splat_ui8() -> tensor<10xui8> {
  %one = "tosa.const"() {values = dense<255> : tensor<10xui8>} : () -> tensor<10xui8>
  %two = "tosa.const"() {values = dense<253> : tensor<10xui8>} : () -> tensor<10xui8>
  // CHECK: "tosa.const"() <{values = dense<2> : tensor<10xui8>}> : () -> tensor<10xui8>
  %sub = tosa.sub %one, %two : (tensor<10xui8>, tensor<10xui8>) -> tensor<10xui8>
  return %sub : tensor<10xui8>
}

// -----

// CHECK-LABEL: @fold_sub_splat_ui8_overflow
func.func @fold_sub_splat_ui8_overflow() -> tensor<10xui8> {
  %one = "tosa.const"() {values = dense<1> : tensor<10xui8>} : () -> tensor<10xui8>
  %two = "tosa.const"() {values = dense<253> : tensor<10xui8>} : () -> tensor<10xui8>
  // CHECK: tosa.sub
  %sub = tosa.sub %one, %two : (tensor<10xui8>, tensor<10xui8>) -> tensor<10xui8>
  return %sub : tensor<10xui8>
}

// -----

// CHECK-LABEL: @fold_greater_splat_f32
func.func @fold_greater_splat_f32() -> (tensor<10xi1>, tensor<10xi1>) {
  %0 = "tosa.const"() {values = dense<4.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %1 = "tosa.const"() {values = dense<2.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %2 = "tosa.const"() {values = dense<1.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %3 = "tosa.const"() {values = dense<2.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %true = tosa.greater %0, %1 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  %false = tosa.greater %2, %3 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  // CHECK-DAG: %[[TRUE:.+]] = "tosa.const"() <{values = dense<true> : tensor<10xi1>}
  // CHECK-DAG: %[[FALSE:.+]] = "tosa.const"() <{values = dense<false> : tensor<10xi1>}
  // CHECK: return %[[TRUE]], %[[FALSE]]
  return %true, %false : tensor<10xi1>, tensor<10xi1>
}

// -----

// CHECK-LABEL: @fold_greater_splat_i32
func.func @fold_greater_splat_i32() -> (tensor<10xi1>, tensor<10xi1>) {
  %0 = "tosa.const"() {values = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %1 = "tosa.const"() {values = dense<8> : tensor<10xi32>} : () -> tensor<10xi32>
  %2 = "tosa.const"() {values = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %3 = "tosa.const"() {values = dense<-12> : tensor<10xi32>} : () -> tensor<10xi32>
  %false = tosa.greater %0, %1 : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  %true = tosa.greater %2, %3 : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  // CHECK-DAG: %[[FALSE:.+]] = "tosa.const"() <{values = dense<false> : tensor<10xi1>}
  // CHECK-DAG: %[[TRUE:.+]] = "tosa.const"() <{values = dense<true> : tensor<10xi1>}
  // CHECK: return %[[FALSE]], %[[TRUE]]
  return %false, %true : tensor<10xi1>, tensor<10xi1>
}

// -----

// CHECK-LABEL: @fold_greater_splat_ui8
func.func @fold_greater_splat_ui8() -> (tensor<10xi1>, tensor<10xi1>, tensor<10xi1>) {
  %0 = "tosa.const"() {values = dense<1> : tensor<10xui8>} : () -> tensor<10xui8>
  %1 = "tosa.const"() {values = dense<1> : tensor<10xui8>} : () -> tensor<10xui8>
  %2 = "tosa.const"() {values = dense<246> : tensor<10xui8>} : () -> tensor<10xui8>
  %3 = "tosa.const"() {values = dense<245> : tensor<10xui8>} : () -> tensor<10xui8>
  %true = tosa.greater %2, %3 : (tensor<10xui8>, tensor<10xui8>) -> tensor<10xi1>
  %false = tosa.greater %0, %1 : (tensor<10xui8>, tensor<10xui8>) -> tensor<10xi1>
  %false2 = tosa.greater %0, %2 : (tensor<10xui8>, tensor<10xui8>) -> tensor<10xi1>
  // CHECK-DAG: %[[TRUE:.+]] = "tosa.const"() <{values = dense<true> : tensor<10xi1>}
  // CHECK-DAG: %[[FALSE:.+]] = "tosa.const"() <{values = dense<false> : tensor<10xi1>}
  // CHECK: return %[[TRUE]], %[[FALSE]], %[[FALSE]]
  return %true, %false, %false2 : tensor<10xi1>, tensor<10xi1>, tensor<10xi1>
}

// -----

// CHECK-LABEL: @fold_greater_eq_splat_f32
func.func @fold_greater_eq_splat_f32() -> (tensor<10xi1>, tensor<10xi1>) {
  %0 = "tosa.const"() {values = dense<4.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %1 = "tosa.const"() {values = dense<4.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %2 = "tosa.const"() {values = dense<1.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %3 = "tosa.const"() {values = dense<2.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %true = tosa.greater_equal %0, %1 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  %false = tosa.greater_equal %2, %3 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  // CHECK-DAG: %[[TRUE:.+]] = "tosa.const"() <{values = dense<true> : tensor<10xi1>}
  // CHECK-DAG: %[[FALSE:.+]] = "tosa.const"() <{values = dense<false> : tensor<10xi1>}
  // CHECK: return %[[TRUE]], %[[FALSE]]
  return %true, %false : tensor<10xi1>, tensor<10xi1>
}

// -----

// CHECK-LABEL: @fold_greater_eq_splat_i32
func.func @fold_greater_eq_splat_i32() -> (tensor<10xi1>, tensor<10xi1>) {
  %0 = "tosa.const"() {values = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %1 = "tosa.const"() {values = dense<8> : tensor<10xi32>} : () -> tensor<10xi32>
  %2 = "tosa.const"() {values = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %3 = "tosa.const"() {values = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %true = tosa.greater_equal %2, %3 : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  %false = tosa.greater_equal %0, %1 : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  // CHECK-DAG: %[[TRUE:.+]] = "tosa.const"() <{values = dense<true> : tensor<10xi1>}
  // CHECK-DAG: %[[FALSE:.+]] = "tosa.const"() <{values = dense<false> : tensor<10xi1>}
  // CHECK: return %[[TRUE]], %[[FALSE]]
  return %true, %false : tensor<10xi1>, tensor<10xi1>
}

// -----

// CHECK-LABEL: @fold_greater_eq_splat_ui8
func.func @fold_greater_eq_splat_ui8() -> (tensor<10xi1>, tensor<10xi1>) {
  %0 = "tosa.const"() {values = dense<1> : tensor<10xui8>} : () -> tensor<10xui8>
  %1 = "tosa.const"() {values = dense<255> : tensor<10xui8>} : () -> tensor<10xui8>
  %2 = "tosa.const"() {values = dense<245> : tensor<10xui8>} : () -> tensor<10xui8>
  %3 = "tosa.const"() {values = dense<245> : tensor<10xui8>} : () -> tensor<10xui8>
  %true = tosa.greater_equal %2, %3 : (tensor<10xui8>, tensor<10xui8>) -> tensor<10xi1>
  %false = tosa.greater_equal %0, %1 : (tensor<10xui8>, tensor<10xui8>) -> tensor<10xi1>
  // CHECK-DAG: %[[TRUE:.+]] = "tosa.const"() <{values = dense<true> : tensor<10xi1>}
  // CHECK-DAG: %[[FALSE:.+]] = "tosa.const"() <{values = dense<false> : tensor<10xi1>}
  // CHECK: return %[[TRUE]], %[[FALSE]]
  return %true, %false : tensor<10xi1>, tensor<10xi1>
}

// -----

// CHECK-LABEL: @fold_eq_splat_f32
func.func @fold_eq_splat_f32() -> (tensor<10xi1>, tensor<10xi1>) {
  %0 = "tosa.const"() {values = dense<4.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %1 = "tosa.const"() {values = dense<4.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %2 = "tosa.const"() {values = dense<1.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %3 = "tosa.const"() {values = dense<2.0> : tensor<10xf32>} : () -> tensor<10xf32>
  %true = tosa.equal %0, %1 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  %false = tosa.equal %2, %3 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  // CHECK-DAG: %[[TRUE:.+]] = "tosa.const"() <{values = dense<true> : tensor<10xi1>}
  // CHECK-DAG: %[[FALSE:.+]] = "tosa.const"() <{values = dense<false> : tensor<10xi1>}
  // CHECK: return %[[TRUE]], %[[FALSE]]
  return %true, %false : tensor<10xi1>, tensor<10xi1>
}

// -----

// CHECK-LABEL: @fold_eq_splat_i32
func.func @fold_eq_splat_i32() -> (tensor<10xi1>, tensor<10xi1>) {
  %0 = "tosa.const"() {values = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %1 = "tosa.const"() {values = dense<8> : tensor<10xi32>} : () -> tensor<10xi32>
  %2 = "tosa.const"() {values = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %3 = "tosa.const"() {values = dense<-10> : tensor<10xi32>} : () -> tensor<10xi32>
  %true = tosa.equal %2, %3 : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  %false = tosa.equal %0, %1 : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  // CHECK-DAG: %[[TRUE:.+]] = "tosa.const"() <{values = dense<true> : tensor<10xi1>}
  // CHECK-DAG: %[[FALSE:.+]] = "tosa.const"() <{values = dense<false> : tensor<10xi1>}
  // CHECK: return %[[TRUE]], %[[FALSE]]
  return %true, %false : tensor<10xi1>, tensor<10xi1>
}

// -----

// CHECK-LABEL: @fold_eq_i32
func.func @fold_eq_i32(%arg0 : tensor<10xi32>) -> (tensor<10xi1>) {
  // CHECK: %[[TRUE:.+]] = "tosa.const"() <{values = dense<true> : tensor<10xi1>}
  %0 = tosa.equal %arg0, %arg0 : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  // CHECK: return %[[TRUE]]
  return %0 : tensor<10xi1>
}

// -----

func.func @reshape_splat() -> tensor<6x5x4xi32> {
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() <{values = dense<42> : tensor<6x5x4xi32>}
  %splat = "tosa.const"() {values = dense<42> : tensor<4x5x6xi32>} : () -> tensor<4x5x6xi32>
  %const = tosa.const_shape {values = dense<[6, 5, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %reshape = tosa.reshape %splat, %const : (tensor<4x5x6xi32>, !tosa.shape<3>) -> tensor<6x5x4xi32>
  // CHECK: return %[[SPLAT]]
  return %reshape : tensor<6x5x4xi32>
}

// -----

// CHECK-LABEL: @slice_splat
func.func @slice_splat() -> tensor<1x1x1xi32> {
  // CHECK: %[[SLICE:.+]] = "tosa.const"() <{values = dense<42> : tensor<1x1x1xi32>}
  %splat = "tosa.const"() {values = dense<42> : tensor<4x5x6xi32>} : () -> tensor<4x5x6xi32>
  %start = tosa.const_shape {values = dense<[1, 2, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %size = tosa.const_shape {values = dense<[1, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %slice= tosa.slice %splat, %start, %size : (tensor<4x5x6xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x1x1xi32>

  // CHECK: return %[[SLICE]]
  return %slice : tensor<1x1x1xi32>
}

// -----

// CHECK-LABEL: @slice_singleton
func.func @slice_singleton() -> tensor<1x1xi32> {
  %splat = "tosa.const"() {values = dense<[[0, 1, 2], [3, 4, 5], [6, 7 ,8]]> : tensor<3x3xi32>} : () -> tensor<3x3xi32>
  // CHECK: %[[SLICE:.+]] = "tosa.const"() <{values = dense<4> : tensor<1x1xi32>}
  %start = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %size = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %slice= tosa.slice %splat, %start, %size : (tensor<3x3xi32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x1xi32>
  // CHECK: return %[[SLICE]]
  return %slice : tensor<1x1xi32>
}

// -----

// CHECK: func.func @cast_float_to_float
func.func @cast_float_to_float() -> tensor<f16> {
  %splat = "tosa.const"() {values = dense<42.0> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() <{values = dense<4.200000e+01> : tensor<f16>}
  %cast = tosa.cast %splat : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[SPLAT]]
  return %cast : tensor<f16>
}

// -----

// CHECK: func.func @cast_int_to_float
func.func @cast_int_to_float() -> tensor<f16> {
  %splat = "tosa.const"() {values = dense<4> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() <{values = dense<4.000000e+00> : tensor<f16>}
  %cast = tosa.cast %splat : (tensor<i32>) -> tensor<f16>
  // CHECK: return %[[SPLAT]]
  return %cast : tensor<f16>
}

// -----

// CHECK: func.func @cast_float_to_int
func.func @cast_float_to_int() -> tensor<i16> {
  %splat = "tosa.const"() {values = dense<-4.0> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() <{values = dense<-4> : tensor<i16>}
  %cast = tosa.cast %splat : (tensor<f32>) -> tensor<i16>
  // CHECK: return %[[SPLAT]]
  return %cast : tensor<i16>
}

// -----

// CHECK: func.func @cast_float_to_int_round
func.func @cast_float_to_int_round() -> tensor<i16> {
  %splat = "tosa.const"() {values = dense<-3.5> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() <{values = dense<-4> : tensor<i16>}
  %cast = tosa.cast %splat : (tensor<f32>) -> tensor<i16>
  // CHECK: return %[[SPLAT]]
  return %cast : tensor<i16>
}

// -----

// CHECK: func.func @cast_int_to_int_trunc
func.func @cast_int_to_int_trunc() -> tensor<i16> {
  %splat = "tosa.const"() {values = dense<-1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() <{values = dense<-1> : tensor<i16>}
  %cast = tosa.cast %splat : (tensor<i32>) -> tensor<i16>
  // CHECK: return %[[SPLAT]]
  return %cast : tensor<i16>
}

// -----

// CHECK: func.func @cast_int_to_int_sign
func.func @cast_int_to_int_sign() -> tensor<i32> {
  %splat = "tosa.const"() {values = dense<-1> : tensor<i16>} : () -> tensor<i16>
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() <{values = dense<-1> : tensor<i32>}
  %cast = tosa.cast %splat : (tensor<i16>) -> tensor<i32>
  // CHECK: return %[[SPLAT]]
  return %cast : tensor<i32>
}

// -----

// CHECK-LABEL: @reverse_splat
func.func @reverse_splat() -> tensor<10xi32> {
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() <{values = dense<42> : tensor<10xi32>}
  %splat = "tosa.const"() {values = dense<42> : tensor<10xi32>} : () -> tensor<10xi32>
  %reverse = tosa.reverse %splat { axis = 0 : i32 } : (tensor<10xi32>) -> tensor<10xi32>
  // CHECK: return %[[SPLAT]]
  return %reverse : tensor<10xi32>
}

// -----

// CHECK-LABEL: @reverse_length_one
func.func @reverse_length_one(%arg0 : tensor<10x1xi32>) -> (tensor<10x1xi32>, tensor<10x1xi32>) {
  %nofold = tosa.reverse %arg0 { axis = 0 : i32 } : (tensor<10x1xi32>) -> tensor<10x1xi32>
  %fold = tosa.reverse %arg0 { axis = 1 : i32 } : (tensor<10x1xi32>) -> tensor<10x1xi32>
  // CHECK: %[[NOFOLD:.+]] = tosa.reverse %arg0 {axis = 0 : i32}
  // CHECK: return %[[NOFOLD]], %arg0
  return %nofold, %fold : tensor<10x1xi32>, tensor<10x1xi32>
}

// -----

// no_shift_op_reorder checks that %arg1 won't be reorder with %0
// by the folder pass.
// CHECK-LABEL: @no_shift_op_reorder
func.func @no_shift_op_reorder (%arg0 : tensor<44x1xi16>, %arg1 : tensor<1xi8>) -> tensor<44x57xi32> {
  %0 = "tosa.const"() {values = dense<1> : tensor<44x57xi16>} : () -> tensor<44x57xi16>
  // CHECK: tosa.mul %arg0, %0, %arg1
  %1 = tosa.mul %arg0, %0, %arg1 : (tensor<44x1xi16>, tensor<44x57xi16>, tensor<1xi8>) -> tensor<44x57xi32>
  return %1 : tensor<44x57xi32>
}

// -----

// CHECK-LABEL: @fold_select_same_value
func.func @fold_select_same_value(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = tosa.select %arg0, %arg1, %arg1 : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: return %arg1
  // CHECK-NOT: tosa.select
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @fold_select_same_value_dynamic_broadcast
func.func @fold_select_same_value_dynamic_broadcast(%arg0: tensor<1x1xi1>, %arg1: tensor<?x3xi32>) -> tensor<?x3xi32> {
  %0 = tosa.select %arg0, %arg1, %arg1 : (tensor<1x1xi1>, tensor<?x3xi32>, tensor<?x3xi32>) -> tensor<?x3xi32>
  // CHECK: return %arg1
  // CHECK-NOT: tosa.select
  return %0 : tensor<?x3xi32>
}

// -----

// CHECK-LABEL: @fold_select_true_value
func.func @fold_select_true_value(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %c1 = "tosa.const"() {values = dense<1> : tensor<2x3xi1>} : () -> tensor<2x3xi1>
  %0 = tosa.select %c1, %arg0, %arg1 : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: return %arg0
  // CHECK-NOT: tosa.select
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @fold_select_false_value
func.func @fold_select_false_value(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %c0 = "tosa.const"() {values = dense<0> : tensor<2x3xi1>} : () -> tensor<2x3xi1>
  %0 = tosa.select %c0, %arg0, %arg1 : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: return %arg1
  // CHECK-NOT: tosa.select
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @no_fold_select_broadcast_same_value
func.func @no_fold_select_broadcast_same_value(%arg0: tensor<2x2xi1>, %arg1: tensor<1x1xf32>) -> tensor<2x2xf32> {
  // CHECK: tosa.select %arg0, %arg1, %arg1
  %0 = tosa.select %arg0, %arg1, %arg1 : (tensor<2x2xi1>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: @no_fold_select_broadcast_true_value
func.func @no_fold_select_broadcast_true_value(%arg0: tensor<1x1xf32>, %arg1: tensor<2x2xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[CONST:.*]] = "tosa.const"
  %0 = "tosa.const"() {values = dense<1> : tensor<2x2xi1>} : () -> tensor<2x2xi1>
  // CHECK: tosa.select %[[CONST]], %arg0, %arg1
  %1 = tosa.select %0, %arg0, %arg1 : (tensor<2x2xi1>, tensor<1x1xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @no_fold_select_broadcast_false_value
func.func @no_fold_select_broadcast_false_value(%arg0: tensor<2x2xf32>, %arg1: tensor<1x1xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[CONST:.*]] = "tosa.const"
  %0 = "tosa.const"() {values = dense<0> : tensor<2x2xi1>} : () -> tensor<2x2xi1>
  // CHECK: tosa.select %[[CONST]], %arg0, %arg1
  %1 = tosa.select %0, %arg0, %arg1 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<1x1xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: @no_fold_select_unknown_broadcast_true_value_dynamic_operand
func.func @no_fold_select_unknown_broadcast_true_value_dynamic_operand(%arg0: tensor<2x?xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x?xf32> {
  // CHECK: %[[CONST:.*]] = "tosa.const"
  %0 = "tosa.const"() {values = dense<1> : tensor<2x2xi1>} : () -> tensor<2x2xi1>
  // CHECK: tosa.select %[[CONST]], %arg0, %arg1
  %1 = tosa.select %0, %arg0, %arg1 : (tensor<2x2xi1>, tensor<2x?xf32>, tensor<2x2xf32>) -> tensor<2x?xf32>
  return %1 : tensor<2x?xf32>
}

// -----

// CHECK-LABEL: @no_fold_select_unknown_broadcast_false_value_dynamic_operand
func.func @no_fold_select_unknown_broadcast_false_value_dynamic_operand(%arg0: tensor<2x?xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[CONST:.*]] = "tosa.const"
  %0 = "tosa.const"() {values = dense<0> : tensor<2x2xi1>} : () -> tensor<2x2xi1>
  // CHECK: tosa.select %[[CONST]], %arg0, %arg1
  %1 = tosa.select %0, %arg0, %arg1 : (tensor<2x2xi1>, tensor<2x?xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: @fold_select_dynamic_broadcast_true_value
func.func @fold_select_dynamic_broadcast_true_value(%arg0: tensor<2x?xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x?xf32> {
  %0 = "tosa.const"() {values = dense<1> : tensor<2x1xi1>} : () -> tensor<2x1xi1>
  %1 = tosa.select %0, %arg0, %arg1 : (tensor<2x1xi1>, tensor<2x?xf32>, tensor<2x1xf32>) -> tensor<2x?xf32>
  // CHECK: return %arg0
  // CHECK-NOT: tosa.select
  return %1 : tensor<2x?xf32>
}

// -----

// CHECK-LABEL: @fold_select_dynamic_broadcast_false_value
func.func @fold_select_dynamic_broadcast_false_value(%arg0: tensor<1x1xf32>, %arg1: tensor<?x2xf32>) -> tensor<?x2xf32> {
  %0 = "tosa.const"() {values = dense<0> : tensor<1x1xi1>} : () -> tensor<1x1xi1>
  %1 = tosa.select %0, %arg0, %arg1 : (tensor<1x1xi1>, tensor<1x1xf32>, tensor<?x2xf32>) -> tensor<?x2xf32>
  // CHECK: return %arg1
  // CHECK-NOT: tosa.select
  return %1 : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: @test_fold_add_shape
// CHECK: tosa.const_shape  {values = dense<[2, 4, 6, 8, 10, 12]> : tensor<6xindex>} : () -> !tosa.shape<6>
func.func @test_fold_add_shape() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[1, 2, 3, 4, 5, 6]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[1, 2, 3, 4, 5, 6]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.add_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_no_fold_add_shape_positive_overflow
// CHECK: tosa.add_shape
func.func @test_no_fold_add_shape_positive_overflow() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[1, 2, 3, 4, 5, 9223372036854775807]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[1, 2, 3, 4, 5, 1]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.add_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_no_fold_add_shape_negative_overflow
// CHECK: tosa.add_shape
func.func @test_no_fold_add_shape_negative_overflow() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[1, 2, 3, 4, 5, -9223372036854775808]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[1, 2, 3, 4, 5, -1]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.add_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_fold_dim
// CHECK: tosa.const_shape  {values = dense<6> : tensor<1xindex>} : () -> !tosa.shape<1>
func.func @test_fold_dim(%arg0: tensor<6xi32>) -> !tosa.shape<1> {
  %dim = tosa.dim %arg0 {axis = 0 : i32} : (tensor<6xi32>) -> !tosa.shape<1>
  return %dim : !tosa.shape<1>
}

// -----

// CHECK-LABEL: @test_fold_sub_shape
// CHECK: tosa.const_shape  {values = dense<[2, 4, 6, 8, 10, 12]> : tensor<6xindex>} : () -> !tosa.shape<6>
func.func @test_fold_sub_shape() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[12, 22, 32, 42, 52, 62]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[10, 18, 26, 34, 42, 50]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.sub_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_no_fold_dim_unranked_input
// CHECK: tosa.dim
func.func @test_no_fold_dim_unranked_input(%arg0: tensor<*xi32>) -> !tosa.shape<1> {
  %dim = tosa.dim %arg0 {axis = 0 : i32} : (tensor<*xi32>) -> !tosa.shape<1>
  return %dim : !tosa.shape<1>
}

// -----

// CHECK-LABEL: @test_no_fold_sub_shape_positive_overflow
// CHECK: tosa.sub_shape
func.func @test_no_fold_sub_shape_positive_overflow() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[1, 2, 3, 4, 5, 9223372036854775807]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[1, 2, 3, 4, 5, -1]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.sub_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_no_fold_dim_dynamic
// CHECK: tosa.dim
func.func @test_no_fold_dim_dynamic(%arg0: tensor<4x?xi32>) -> !tosa.shape<1> {
  %dim = tosa.dim %arg0 {axis = 1 : i32} : (tensor<4x?xi32>) -> !tosa.shape<1>
  return %dim : !tosa.shape<1>
}

// -----

// CHECK-LABEL: @test_no_fold_sub_shape_negative_overflow
// CHECK: tosa.sub_shape
func.func @test_no_fold_sub_shape_negative_overflow() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[1, 2, 3, 4, 5, -9223372036854775808]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[1, 2, 3, 4, 5, 1]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.sub_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_fold_mul_shape
// CHECK: tosa.const_shape  {values = dense<[2, 4, 6, 8, 10, 12]> : tensor<6xindex>} : () -> !tosa.shape<6>
func.func @test_fold_mul_shape() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[2, 2, 3, 4, 2, 3]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[1, 2, 2, 2, 5, 4]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.mul_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_no_fold_mul_shape_positive_overflow
// CHECK: tosa.mul_shape
func.func @test_no_fold_mul_shape_positive_overflow() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[1, 2, 3, 4, 5, 9223372036854775807]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[1, 2, 3, 4, 5, 2]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.mul_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_no_fold_mul_shape_negative_overflow
// CHECK: tosa.mul_shape
func.func @test_no_fold_mul_shape_negative_overflow() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[1, 2, 3, 4, 5, -9223372036854775808]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[1, 2, 3, 4, 5, 2]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.mul_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_fold_div_ceil_shape
// CHECK: tosa.const_shape  {values = dense<[2, 4, 6, 8, 10, 12]> : tensor<6xindex>} : () -> !tosa.shape<6>
func.func @test_fold_div_ceil_shape() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[2, 7, 11, 22, 47, 46]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[1, 2, 2, 3, 5, 4]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.div_ceil_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_no_fold_div_ceil_shape_positive_overflow
// CHECK: tosa.div_ceil_shape
func.func @test_no_fold_div_ceil_shape_positive_overflow() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[2, 7, 11, 22, 47, 9223372036854775807]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[1, 2, 2, 3, 5, 0]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.div_ceil_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_no_fold_div_ceil_shape_negative_overflow
// CHECK: tosa.div_ceil_shape
func.func @test_no_fold_div_ceil_shape_negative_overflow() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[2, 7, 11, 22, 47, -9223372036854775808]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[1, 2, 2, 3, 5, 0]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.div_ceil_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_fold_div_floor_shape
// CHECK: tosa.const_shape  {values = dense<[2, 3, 5, 7, 9, 11]> : tensor<6xindex>} : () -> !tosa.shape<6>
func.func @test_fold_div_floor_shape() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[2, 7, 11, 22, 47, 46]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[1, 2, 2, 3, 5, 4]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.div_floor_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_no_fold_div_floor_shape_positive_overflow
// CHECK: tosa.div_floor_shape
func.func @test_no_fold_div_floor_shape_positive_overflow() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[2, 7, 11, 22, 47, 9223372036854775807]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[1, 2, 2, 3, 5, 0]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.div_floor_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_no_fold_div_floor_shape_negative_overflow
// CHECK: tosa.div_floor_shape
func.func @test_no_fold_div_floor_shape_negative_overflow() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[2, 7, 11, 22, 47, -9223372036854775808]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[1, 2, 2, 3, 5, 0]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.div_floor_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_fold_mod_shape
// CHECK: tosa.const_shape  {values = dense<[2, 1, 5, 7, 9, 11]> : tensor<6xindex>} : () -> !tosa.shape<6>
func.func @test_fold_mod_shape() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[24, 7, 65, 33, 39, 39]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[11, 2, 12, 13, 15, 14]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.mod_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_no_fold_mod_shape_positive_overflow
// CHECK: tosa.mod_shape
func.func @test_no_fold_mod_shape_positive_overflow() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[24, 7, 65, 33, 39, 9223372036854775807]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[11, 2, 12, 13, 15, 0]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.mod_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_no_fold_mod_shape_negative_overflow
// CHECK: tosa.mod_shape
func.func @test_no_fold_mod_shape_negative_overflow() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[24, 7, 65, 33, 39, -9223372036854775808]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[11, 2, 12, 13, 15, 0]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.mod_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_max_shape
// CHECK: tosa.const_shape  {values = dense<[24, 7, 65, 33, 39, 5]> : tensor<6xindex>} : () -> !tosa.shape<6>
func.func @test_max_shape() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[24, 7, 65, 33, 39, 1]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[11, 2, 12, 13, 15, 5]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.max_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_min_shape
// CHECK: tosa.const_shape  {values = dense<[11, 2, 12, 13, 15, 1]> : tensor<6xindex>} : () -> !tosa.shape<6>
func.func @test_min_shape() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[24, 7, 65, 33, 39, 1]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %b = tosa.const_shape {values = dense<[11, 2, 12, 13, 15, 5]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.min_shape %a, %b : (!tosa.shape<6>, !tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_exp2_shape
// CHECK: tosa.const_shape  {values = dense<[4, 8, 2, 16, 64, 32]> : tensor<6xindex>} : () -> !tosa.shape<6>
func.func @test_exp2_shape() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[2, 3, 1, 4, 6, 5]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.exp2_shape %a : (!tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_neg_exp2_shape
// CHECK: tosa.exp2_shape
func.func @test_neg_exp2_shape() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[-10, 3, 1, 4, 6, 5]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.exp2_shape %a : (!tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_high_exp2_shape
// CHECK: tosa.exp2_shape
func.func @test_high_exp2_shape() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[32, 64, 1, 4, 6, 5]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.exp2_shape %a : (!tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_log2_ceil_shape
// CHECK: tosa.const_shape  {values = dense<[2, 4, 4, 0, 3, 5]> : tensor<6xindex>} : () -> !tosa.shape<6>
func.func @test_log2_ceil_shape() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[4, 9, 14, 1, 7, 30]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.log2_ceil_shape %a : (!tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_log2_ceil_shape_zero
// CHECK: tosa.log2_ceil_shape
func.func @test_log2_ceil_shape_zero() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[4, 9, 0, 1, 7, 30]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.log2_ceil_shape %a : (!tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_log2_ceil_shape_neg
// CHECK: tosa.log2_ceil_shape
func.func @test_log2_ceil_shape_neg() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[4, 9, -123, 1, 7, 30]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.log2_ceil_shape %a : (!tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_log2_floor_shape
// CHECK: tosa.const_shape  {values = dense<[2, 3, 3, 0, 2, 4]> : tensor<6xindex>} : () -> !tosa.shape<6>
func.func @test_log2_floor_shape() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[4, 9, 14, 1, 7, 30]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.log2_floor_shape %a : (!tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_log2_floor_shape_zero
// CHECK: tosa.log2_floor_shape
func.func @test_log2_floor_shape_zero() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[4, 9, 0, 1, 7, 30]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.log2_floor_shape %a : (!tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}

// -----

// CHECK-LABEL: @test_log2_floor_shape_neg
// CHECK: tosa.log2_floor_shape
func.func @test_log2_floor_shape_neg() -> !tosa.shape<6> {
  %a = tosa.const_shape {values = dense<[4, 9, -123, 1, 7, 30]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %c = tosa.log2_floor_shape %a : (!tosa.shape<6>) -> !tosa.shape<6>
  return %c : !tosa.shape<6>
}
