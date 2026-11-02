// RUN: not mlir-opt %s --pass-pipeline='builtin.module(any(acc-bind-routine))' 2>&1 | FileCheck %s

module {
  acc.routine @r func(@wrapped) seq bind("actual_impl")
  func.func private @wrapped()
      attributes {acc.routine_info = #acc.routine_info<[@r]>}
  func.func @entry() {
    acc.serial {
      func.call @wrapped() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: error: string-bound ACC routine target was not materialized
