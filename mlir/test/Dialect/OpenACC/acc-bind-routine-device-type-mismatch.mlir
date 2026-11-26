// acc-materialize-routine-bind-targets only declares the bind target selected
// for its device type. When acc-bind-routine is run with a different device
// type, the missing target is reported with the device type it was selected
// for.

// RUN: not mlir-opt %s --pass-pipeline='builtin.module(acc-materialize-routine-bind-targets, any(acc-bind-routine{device-type=nvidia}))' 2>&1 | FileCheck %s

module {
  acc.routine @r func(@wrapped) seq bind("impl_default", "impl_nvidia" [#acc.device_type<nvidia>])
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

// CHECK: error: string-bound ACC routine target 'impl_nvidia' selected for device type 'nvidia' was not materialized, run acc-materialize-routine-bind-targets with the same device-type first
