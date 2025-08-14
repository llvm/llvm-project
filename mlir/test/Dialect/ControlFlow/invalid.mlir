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
