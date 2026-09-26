// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: @assert
func.func @assert(%arg : i1) {
  cf.assert %arg, "Some message in case this assertion fails."
  return
}

// CHECK-LABEL: func @switch(
// CHECK: -1:
func.func @switch(%flag : i32, %caseOperand : i32) {
  cf.switch %flag : i32, [
    default: ^bb1(%caseOperand : i32),
    42: ^bb2(%caseOperand : i32),
    43: ^bb3(%caseOperand : i32),
    -1: ^bb4(%caseOperand : i32)
  ]

  ^bb1(%bb1arg : i32):
    return
  ^bb2(%bb2arg : i32):
    return
  ^bb3(%bb3arg : i32):
    return
  ^bb4(%bb4arg : i32):
    return
}

// CHECK-LABEL: func @switch_i64(
func.func @switch_i64(%flag : i64, %caseOperand : i32) {
  cf.switch %flag : i64, [
    default: ^bb1(%caseOperand : i32),
    42: ^bb2(%caseOperand : i32),
    43: ^bb3(%caseOperand : i32)
  ]

  ^bb1(%bb1arg : i32):
    return
  ^bb2(%bb2arg : i32):
    return
  ^bb3(%bb3arg : i32):
    return
}

// CHECK-LABEL: func @switch_i1_unsigned_boundary(
// CHECK: -1:
func.func @switch_i1_unsigned_boundary(%flag : i1) {
  cf.switch %flag : i1, [
    default: ^bb1,
    1: ^bb1
  ]

  ^bb1:
    return
}

// CHECK-LABEL: func @switch_i8_signed_boundaries(
// CHECK: -128:
// CHECK: 127:
func.func @switch_i8_signed_boundaries(%flag : i8) {
  cf.switch %flag : i8, [
    default: ^bb1,
    -128: ^bb1,
    127: ^bb1
  ]

  ^bb1:
    return
}

// CHECK-LABEL: func @switch_i8_unsigned_boundaries(
// CHECK: -128:
// CHECK: -1:
func.func @switch_i8_unsigned_boundaries(%flag : i8) {
  cf.switch %flag : i8, [
    default: ^bb1,
    128: ^bb1,
    255: ^bb1
  ]

  ^bb1:
    return
}

// CHECK-LABEL: func @switch_i64_signed_boundaries(
// CHECK: -9223372036854775808:
// CHECK: 9223372036854775807:
func.func @switch_i64_signed_boundaries(%flag : i64) {
  cf.switch %flag : i64, [
    default: ^bb1,
    -9223372036854775808: ^bb1,
    9223372036854775807: ^bb1
  ]

  ^bb1:
    return
}

// CHECK-LABEL: func @switch_i64_unsigned_boundary(
// CHECK: -1:
func.func @switch_i64_unsigned_boundary(%flag : i64) {
  cf.switch %flag : i64, [
    default: ^bb1,
    18446744073709551615: ^bb1
  ]

  ^bb1:
    return
}

// CHECK-LABEL: func @switch_i128_signed_boundaries(
// CHECK: -170141183460469231731687303715884105728:
// CHECK: 170141183460469231731687303715884105727:
func.func @switch_i128_signed_boundaries(%flag : i128) {
  cf.switch %flag : i128, [
    default: ^bb1,
    -170141183460469231731687303715884105728: ^bb1,
    170141183460469231731687303715884105727: ^bb1
  ]

  ^bb1:
    return
}

// CHECK-LABEL: func @switch_i128_unsigned_boundaries(
// CHECK: -170141183460469231731687303715884105728:
// CHECK: -1:
func.func @switch_i128_unsigned_boundaries(%flag : i128) {
  cf.switch %flag : i128, [
    default: ^bb1,
    170141183460469231731687303715884105728: ^bb1,
    340282366920938463463374607431768211455: ^bb1
  ]

  ^bb1:
    return
}

// CHECK-LABEL: func @switch_result_number
func.func @switch_result_number(%arg0: i32) {
  %0:2 = "test.op_with_two_results"() : () -> (i32, i32)
  cf.switch %arg0 : i32, [
    default: ^bb2,
    0: ^bb1(%0#0 : i32)
  ]
  ^bb1(%1: i32):
    return
  ^bb2:
    return
}

// CHECK-LABEL: func @switch_result_number_default
func.func @switch_result_number_default(%arg0: i32) {
  %0:2 = "test.op_with_two_results"() : () -> (i32, i32)
  cf.switch %arg0 : i32, [
    default: ^bb1(%0#0 : i32),
    0: ^bb2(%0#1 : i32)
  ]
  ^bb1(%1: i32):
    return
  ^bb2(%2: i32):
    return
}

// CHECK-LABEL: func @cond_weights
func.func @cond_weights(%cond: i1) {
// CHECK: cf.cond_br %{{.*}} weights([60, 40]), ^{{.*}}, ^{{.*}}
  cf.cond_br %cond weights([60, 40]), ^bb1, ^bb2
  ^bb1:
    return
  ^bb2:
    return
}
