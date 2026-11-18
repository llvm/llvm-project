// RUN: mlir-opt %s --pass-pipeline='builtin.module(acc-materialize-routine-bind-targets{device-type=nvidia}, any(acc-bind-routine{device-type=nvidia}))' -split-input-file | FileCheck %s

// The device_type(*) bind applies as target device type NVIDIA isn't named in the directive
module {
  acc.routine @r_star func(@wrapped_star) seq bind("impl_star" [#acc.device_type<star>])
  func.func private @wrapped_star()
      attributes {acc.routine_info = #acc.routine_info<[@r_star]>}
  func.func @star_only() {
    acc.serial {
      func.call @wrapped_star() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @star_only()
// CHECK: func.call @impl_star() : () -> ()
// CHECK: func.func private @impl_star()

// -----

// The device_type(*) bind applies as target device type NVIDIA isn't named in the directive
module {
  acc.routine @r_star_default func(@wrapped_star_default) seq bind("impl_default", "impl_star" [#acc.device_type<star>])
  func.func private @wrapped_star_default()
      attributes {acc.routine_info = #acc.routine_info<[@r_star_default]>}
  func.func @star_over_default() {
    acc.serial {
      func.call @wrapped_star_default() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @star_over_default()
// CHECK: func.call @impl_star() : () -> ()
// CHECK: func.func private @impl_star()

// -----

// The default bind applies as target device type NVIDIA is named but does not have a bind
module {
  acc.routine @r_named func(@wrapped_named) seq([#acc.device_type<nvidia>]) bind("impl_default", "impl_star" [#acc.device_type<star>])
  func.func private @wrapped_named()
      attributes {acc.routine_info = #acc.routine_info<[@r_named]>}
  func.func @named_device_default() {
    acc.serial {
      func.call @wrapped_named() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @named_device_default()
// CHECK: func.call @impl_default() : () -> ()
// CHECK: func.func private @impl_default()

// -----

// The NVIDIA-specific bind applies as target device type NVIDIA is named and has a bind
module {
  acc.routine @r_named func(@wrapped_named) seq bind("impl_default", "impl_nvidia" [#acc.device_type<nvidia>])
  func.func private @wrapped_named()
      attributes {acc.routine_info = #acc.routine_info<[@r_named]>}
  func.func @named_device_bind() {
    acc.serial {
      func.call @wrapped_named() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @named_device_bind()
// CHECK: func.call @impl_nvidia() : () -> ()
// CHECK: func.func private @impl_nvidia()
