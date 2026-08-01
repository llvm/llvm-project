// RUN: mlir-opt %s -acc-to-llvm -verify-diagnostics -split-input-file | FileCheck %s

// A declaration of a runtime symbol with an incompatible signature cannot be
// called with the arguments the conversion emits.
module {
  llvm.func @__tgt_acc_wait(!llvm.ptr, i64, i64, i64) -> i32
  func.func @mismatched_wait() {
    // expected-error @below {{OpenACC runtime function '__tgt_acc_wait' is already declared with signature}}
    // expected-error @below {{failed to legalize operation 'acc.wait'}}
    acc.wait
    return
  }
}

// -----

// A matching declaration is reused.
// CHECK-LABEL: llvm.func @matching_init
// CHECK: llvm.call @__tgt_acc_init
module {
  llvm.func @__tgt_acc_init(!llvm.ptr, i64, i64, i64)
  func.func @matching_init() {
    acc.init
    return
  }
}
