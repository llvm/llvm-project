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

// -----

// Data mapping calls use the same checked declaration path.
module {
  llvm.func @__tgt_acc_data_begin(!llvm.ptr, i64)
  func.func @mismatched_data(%arg0: !llvm.ptr) {
    %size = arith.constant 4 : i64
    // The mapping call is reported at the operand that states the mapping,
    // which is where the runtime arguments are emitted.
    // expected-error @below {{OpenACC runtime function '__tgt_acc_data_begin' is already declared with signature}}
    %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
        size(%size : i64) elementSize(4) descKind(none)
        mapFlags(to) -> !llvm.ptr
    // expected-error @below {{failed to legalize operation 'acc.data'}}
    acc.data dataOperands(%map : !llvm.ptr) {
      acc.terminator
    }
    return
  }
}
