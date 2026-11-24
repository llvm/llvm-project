// RUN: mlir-opt --lift-cf-to-scf -verify-diagnostics -split-input-file %s

// Verify that --lift-cf-to-scf does not crash when it encounters an
// unreachable blocks (dead code).  Instead it should emit a clean error.
// See: https://github.com/llvm/llvm-project/issues/206086

module {
  func.func @single_return_statement(%arg0: i32) -> i32 {
    return %arg0 : i32
  ^bb1:  // pred: ^bb1
    // expected-error@below {{'arith.constant' op transformation does not support unreachable blocks}}
    %true = arith.constant true
    cf.cond_br %true, ^bb2, ^bb1
  ^bb2:  // pred: ^bb1
    return %arg0 : i32
  }
}

