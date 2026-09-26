// RUN: mlir-opt -verify-diagnostics -split-input-file %s

func.func @switch_missing_case_value(%flag : i32, %caseOperand : i32) {
  cf.switch %flag : i32, [
    default: ^bb1(%caseOperand : i32),
    45: ^bb2(%caseOperand : i32),
    // expected-error@+1 {{expected integer value}}
    : ^bb3(%caseOperand : i32)
  ]

  ^bb1(%bb1arg : i32):
    return
  ^bb2(%bb2arg : i32):
    return
  ^bb3(%bb3arg : i32):
    return
}

// -----

func.func @switch_wrong_type_case_value(%flag : i32, %caseOperand : i32) {
  cf.switch %flag : i32, [
    default: ^bb1(%caseOperand : i32),
    // expected-error@+1 {{expected integer value}}
    "hello": ^bb2(%caseOperand : i32)
  ]

  ^bb1(%bb1arg : i32):
    return
  ^bb2(%bb2arg : i32):
    return
  ^bb3(%bb3arg : i32):
    return
}

// -----

func.func @switch_missing_comma(%flag : i32, %caseOperand : i32) {
  cf.switch %flag : i32, [
    default: ^bb1(%caseOperand : i32),
    // expected-error@+1 {{expected ']'}}
    45: ^bb2(%caseOperand : i32)
    43: ^bb3(%caseOperand : i32)
  ]

  ^bb1(%bb1arg : i32):
    return
  ^bb2(%bb2arg : i32):
    return
  ^bb3(%bb3arg : i32):
    return
}

// -----

func.func @switch_missing_default(%flag : i32, %caseOperand : i32) {
  cf.switch %flag : i32, [
    // expected-error@+1 {{expected 'default'}}
    45: ^bb2(%caseOperand : i32)
    43: ^bb3(%caseOperand : i32)
  ]

  ^bb1(%bb1arg : i32):
    return
  ^bb2(%bb2arg : i32):
    return
  ^bb3(%bb3arg : i32):
    return
}

// -----

func.func @switch_i1_positive_overflow(%flag : i1) {
  cf.switch %flag : i1, [
    default: ^bb1,
    // expected-error@+1 {{case value does not fit in flag type 'i1'}}
    2: ^bb1
  ]
^bb1:
  return
}

// -----

func.func @switch_i1_negative_overflow(%flag : i1) {
  cf.switch %flag : i1, [
    default: ^bb1,
    // expected-error@+1 {{case value does not fit in flag type 'i1'}}
    -2: ^bb1
  ]
^bb1:
  return
}

// -----

func.func @switch_i8_positive_overflow(%flag : i8) {
  cf.switch %flag : i8, [
    default: ^bb1,
    // expected-error@+1 {{case value does not fit in flag type 'i8'}}
    256: ^bb1
  ]
^bb1:
  return
}

// -----

func.func @switch_i8_negative_overflow(%flag : i8) {
  cf.switch %flag : i8, [
    default: ^bb1,
    // expected-error@+1 {{case value does not fit in flag type 'i8'}}
    -129: ^bb1
  ]
^bb1:
  return
}

// -----

func.func @switch_i64_positive_overflow(%flag : i64) {
  cf.switch %flag : i64, [
    default: ^bb1,
    // expected-error@+1 {{case value does not fit in flag type 'i64'}}
    18446744073709551616: ^bb1
  ]
^bb1:
  return
}

// -----

func.func @switch_i64_negative_overflow(%flag : i64) {
  cf.switch %flag : i64, [
    default: ^bb1,
    // expected-error@+1 {{case value does not fit in flag type 'i64'}}
    -9223372036854775809: ^bb1
  ]
^bb1:
  return
}

// -----

func.func @switch_i128_positive_overflow(%flag : i128) {
  cf.switch %flag : i128, [
    default: ^bb1,
    // expected-error@+1 {{case value does not fit in flag type 'i128'}}
    340282366920938463463374607431768211456: ^bb1
  ]
^bb1:
  return
}

// -----

func.func @switch_i128_negative_overflow(%flag : i128) {
  cf.switch %flag : i128, [
    default: ^bb1,
    // expected-error@+1 {{case value does not fit in flag type 'i128'}}
    -170141183460469231731687303715884105729: ^bb1
  ]
^bb1:
  return
}

// -----

// CHECK-LABEL: func @wrong_weights_number
func.func @wrong_weights_number(%cond: i1) {
  // expected-error@+1 {{expects number of branch weights to match number of successors: 1 vs 2}}
  cf.cond_br %cond weights([100]), ^bb1, ^bb2
  ^bb1:
    return
  ^bb2:
    return
}

// -----

// CHECK-LABEL: func @zero_weights
func.func @wrong_total_weight(%cond: i1) {
  // expected-error@+1 {{branch weights cannot all be zero}}
  cf.cond_br %cond weights([0, 0]), ^bb1, ^bb2
  ^bb1:
    return
  ^bb2:
    return
}
