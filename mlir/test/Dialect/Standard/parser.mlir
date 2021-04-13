// RUN: mlir-opt -verify-diagnostics -split-input-file %s

func @switch_missing_case_value(%flag : i32, %caseOperand : i32) {
  switch %flag : i32, [
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

func @switch_wrong_type_case_value(%flag : i32, %caseOperand : i32) {
  switch %flag : i32, [
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

func @switch_missing_comma(%flag : i32, %caseOperand : i32) {
  switch %flag : i32, [
    default: ^bb1(%caseOperand : i32),
    45: ^bb2(%caseOperand : i32)
    // expected-error@+1 {{expected ']'}}
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

func @switch_missing_default(%flag : i32, %caseOperand : i32) {
  switch %flag : i32, [
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
