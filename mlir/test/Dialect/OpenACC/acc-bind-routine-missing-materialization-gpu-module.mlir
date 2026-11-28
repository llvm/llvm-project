// acc-materialize-routine-bind-targets does not descend into nested symbol
// tables. Running it only on the outer module leaves the target undeclared in
// the gpu.module where acc-bind-routine looks it up, and the error names that
// gpu.module.

// RUN: not mlir-opt %s --pass-pipeline='builtin.module(acc-materialize-routine-bind-targets, gpu.module(any(acc-bind-routine)))' 2>&1 | FileCheck %s

module attributes {gpu.container_module} {
  acc.routine @r func(@wrapped) seq bind("impl")
  func.func private @wrapped()
      attributes {acc.routine_info = #acc.routine_info<[@r]>}
  gpu.module @kernels {
    func.func private @wrapped()
        attributes {acc.routine_info = #acc.routine_info<[@r]>}
    gpu.func @kernel() kernel {
      func.call @wrapped() : () -> ()
      gpu.return
    }
  }
}

// CHECK: error: string-bound ACC routine target 'impl' selected for device type 'none' was not materialized in the enclosing 'gpu.module' @kernels
// CHECK: note: run acc-materialize-routine-bind-targets{device-type=none} on this 'gpu.module' @kernels before acc-bind-routine
