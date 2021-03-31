// RUN: mlir-opt --split-input-file --tosa-to-standard %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: func @const_test
func @const_test() -> (tensor<i32>) {
  // CHECK: [[C3:%.+]] = constant dense<3> : tensor<i32>
  %0 = "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>

  // CHECK: return [[C3]]
  return %0 : tensor<i32>
}

// -----

func @slice(%arg0: tensor<6xf32>) ->() {
  // CHECK: [[SLICE:%.+]] = subtensor %arg0[2] [1] [1]
  %0 = "tosa.slice"(%arg0) {start = [2], size = [1]} : (tensor<6xf32>)  -> (tensor<1xf32>)
  return
}

// -----

func @apply_scale_test(%arg0 : i32, %arg1 : i32, %arg2 : i8) -> (i32) {
  // CHECK: [[C1_8:%.+]] = constant 1 : i8
  // CHECK: [[C1_32:%.+]] = constant 1 : i32
  // CHECK: [[C1_64:%.+]] = constant 1 : i64
  // CHECK: [[SHIFT_MINUS_ONE_8:%.+]] = subi %arg2, [[C1_8]]

  // CHECK: [[SHIFT_32:%.+]] = sexti %arg2 : i8 to i32
  // CHECK: [[SHIFT_MINUS_ONE_64:%.+]] = sexti [[SHIFT_MINUS_ONE_8]] : i8 to i64
  // CHECK: [[SHIFTED_64:%.+]] = shift_left [[C1_64]], [[SHIFT_MINUS_ONE_64]]

  // CHECK: [[C0_32:%.+]] = constant 0 : i32
  // CHECK: [[C30_32:%.+]] = constant 30 : i32
  // CHECK: [[SECOND_BIAS:%.+]] = shift_left [[C1_32]], [[C30_32]]
  // CHECK: [[SECOND_BIAS_64:%.+]] = sexti [[SECOND_BIAS]] : i32 to i64
  // CHECK: [[POSITIVE_ROUND:%.+]] = addi [[SHIFTED_64]], [[SECOND_BIAS_64]]
  // CHECK: [[NEGATIVE_ROUND:%.+]] = subi [[SHIFTED_64]], [[SECOND_BIAS_64]]
  // CHECK: [[VALUE_NEGATIVE:%.+]] = cmpi sge, %arg0, [[C0_32]] : i32
  // CHECK: [[DOUBLE_ROUNDED:%.+]] = select [[VALUE_NEGATIVE]], [[POSITIVE_ROUND]], [[NEGATIVE_ROUND]] : i64
  // CHECK: [[C32_32:%.+]] = constant 32 : i32
  // CHECK: [[IS_32BIT_SHIFT:%.+]] = cmpi sge, [[SHIFT_32]], [[C32_32]]
  // CHECK: [[ROUND:%.+]] = select [[IS_32BIT_SHIFT]], [[DOUBLE_ROUNDED]], [[SHIFTED_64]]

  // CHECK: [[VAL_64:%.+]] = sexti %arg0 : i32 to i64
  // CHECK: [[MULTIPLY_64:%.+]] = sexti %arg1 : i32 to i64
  // CHECK: [[SHIFT_64:%.+]] = sexti %arg2 : i8 to i64
  // CHECK: [[SCALED:%.+]] = muli [[VAL_64]], [[MULTIPLY_64]]
  // CHECK: [[BIASED:%.+]] = addi [[SCALED]], [[ROUND]]
  // CHECK: [[DOWNSHIFTED:%.+]] = shift_right_signed [[BIASED]], [[SHIFT_64]]
  // CHECK: [[TRUNCATED:%.+]] = trunci [[DOWNSHIFTED]]

  %0 = "tosa.apply_scale"(%arg0, %arg1, %arg2) {double_round = true} : (i32, i32, i8) -> i32
  return %0 : i32
}
