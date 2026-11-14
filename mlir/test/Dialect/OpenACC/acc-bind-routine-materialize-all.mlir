// acc-materialize-routine-bind-targets declares every string bind target, so
// acc-bind-routine can select a target for any device type without the two
// passes being configured with the same device type.

// RUN: mlir-opt %s --pass-pipeline='builtin.module(acc-materialize-routine-bind-targets, any(acc-bind-routine{device-type=nvidia}))' -split-input-file | FileCheck %s --check-prefix=NVIDIA
// RUN: mlir-opt %s --pass-pipeline='builtin.module(acc-materialize-routine-bind-targets, any(acc-bind-routine))' -split-input-file | FileCheck %s --check-prefix=DEFAULT

module {
  acc.routine @r func(@wrapped) seq bind("impl_default", "impl_nvidia" [#acc.device_type<nvidia>])
  func.func private @wrapped()
      attributes {acc.routine_info = #acc.routine_info<[@r]>}
  func.func @default_and_nvidia() {
    acc.serial {
      func.call @wrapped() : () -> ()
      acc.yield
    }
    return
  }
}

// NVIDIA-LABEL: func.func @default_and_nvidia()
// NVIDIA:         func.call @impl_nvidia() : () -> ()
// NVIDIA-DAG:   func.func private @impl_default()
// NVIDIA-DAG:   func.func private @impl_nvidia()

// DEFAULT-LABEL: func.func @default_and_nvidia()
// DEFAULT:         func.call @impl_default() : () -> ()
// DEFAULT-DAG:   func.func private @impl_default()
// DEFAULT-DAG:   func.func private @impl_nvidia()

// -----

// With only a device-specific bind, a call compiled for another device type
// keeps its original callee.

module {
  acc.routine @r func(@wrapped) seq bind("impl_nvidia" [#acc.device_type<nvidia>])
  func.func private @wrapped()
      attributes {acc.routine_info = #acc.routine_info<[@r]>}
  func.func @nvidia_only() {
    acc.serial {
      func.call @wrapped() : () -> ()
      acc.yield
    }
    return
  }
}

// NVIDIA-LABEL: func.func @nvidia_only()
// NVIDIA:         func.call @impl_nvidia() : () -> ()

// DEFAULT-LABEL: func.func @nvidia_only()
// DEFAULT:         func.call @wrapped() : () -> ()

// -----

// Targets of routines that are never called from an offload region are not
// declared.

module {
  acc.routine @r func(@wrapped) seq bind("impl_default", "impl_nvidia" [#acc.device_type<nvidia>])
  func.func private @wrapped()
      attributes {acc.routine_info = #acc.routine_info<[@r]>}
  func.func @host_call() {
    func.call @wrapped() : () -> ()
    return
  }
}

// NVIDIA-LABEL: func.func @host_call()
// NVIDIA:         call @wrapped() : () -> ()
// NVIDIA-NOT:   func.func private @impl_

// DEFAULT-LABEL: func.func @host_call()
// DEFAULT:         call @wrapped() : () -> ()
// DEFAULT-NOT:   func.func private @impl_
