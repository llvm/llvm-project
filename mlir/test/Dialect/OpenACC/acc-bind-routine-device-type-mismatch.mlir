// acc-materialize-routine-bind-targets only declares the bind target selected
// for its device type. When acc-bind-routine needs a different one, the error
// names the target and the device type it was selected for.

// RUN: not mlir-opt %s --pass-pipeline='builtin.module(acc-materialize-routine-bind-targets{device-type=none}, any(acc-bind-routine{device-type=nvidia}))' 2>&1 | FileCheck %s

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

// CHECK: error: string-bound ACC routine target 'impl_nvidia' selected for device type 'nvidia' was not materialized in the enclosing 'builtin.module'
// CHECK: note: run acc-materialize-routine-bind-targets{device-type=nvidia} on this 'builtin.module' before acc-bind-routine

