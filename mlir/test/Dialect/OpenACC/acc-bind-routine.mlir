// RUN: mlir-opt %s --pass-pipeline='builtin.module(acc-materialize-routine-bind-targets, gpu.module(acc-materialize-routine-bind-targets, any(acc-bind-routine)), any(acc-bind-routine))' -split-input-file | FileCheck %s

// Call to routine with bind is rewritten to the bound symbol inside
// offload region.
module {
  acc.routine @r_bind func(@foo) seq bind(@bar)
  func.func @foo() attributes {acc.routine_info = #acc.routine_info<[@r_bind]>} {
    return
  }
  func.func @bar() {
    return
  }
  func.func @main() {
    acc.serial {
      func.call @foo() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: acc.routine @r_bind func(@foo) bind(@bar) seq
// CHECK: func.call @bar() : () -> ()

// -----

// Bind with string name: call is rewritten to the string symbol.
module {
  acc.routine @r_bind_str func(@wrapped) seq bind("actual_impl")
  func.func @wrapped() attributes {acc.routine_info = #acc.routine_info<[@r_bind_str]>} {
    return
  }
  func.func @entry() {
    acc.serial {
      func.call @wrapped() : () -> ()
      acc.yield
    }
    return
  }
  func.func @entry2() {
    acc.serial {
      func.call @wrapped() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @entry
// CHECK: func.call @actual_impl() : () -> ()
// CHECK-LABEL: func.func @entry2
// CHECK: func.call @actual_impl() : () -> ()
// CHECK: func.func private @actual_impl()

// -----

// Call outside offload region is unchanged.
module {
  acc.routine @r_bind func(@foo) seq bind(@bar)
  func.func @foo() attributes {acc.routine_info = #acc.routine_info<[@r_bind]>} {
    return
  }
  func.func @bar() {
    return
  }
  func.func @main() {
    func.call @foo() : () -> ()
    acc.serial {
      func.call @foo() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: call @foo() : () -> ()
// CHECK: func.call @bar() : () -> ()

// -----

// Indirect call (callee is value) is skipped; no crash.
module {
  acc.routine @r_bind func(@target) seq bind(@bound)
  func.func @target() attributes {acc.routine_info = #acc.routine_info<[@r_bind]>} {
    return
  }
  func.func @bound() {
    return
  }
  func.func @caller(%callee: () -> ()) {
    acc.serial {
      func.call_indirect %callee() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: func.call_indirect %{{.*}}() : () -> ()

// -----

module {
  acc.routine @acc_routine_0 func(@my_device_func) bind("__wrapper_my_device_func") seq
  acc.routine @acc_routine_1 func(@my_device_sub) bind("__wrapper_my_device_sub") seq
  func.func private @my_device_func(i32) -> i32 attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}
  func.func private @my_device_sub(i32, memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_1]>}
  gpu.module @cuda_device_mod {
    gpu.func @test(%arg0: i32, %arg1: memref<i32>) {
      %0 = func.call @my_device_func(%arg0) : (i32) -> i32
      func.call @my_device_sub(%arg0, %arg1) : (i32, memref<i32>) -> ()
      gpu.return
    }
    gpu.func @test2(%arg0: i32, %arg1: memref<i32>) {
      %0 = func.call @my_device_func(%arg0) : (i32) -> i32
      func.call @my_device_sub(%arg0, %arg1) : (i32, memref<i32>) -> ()
      gpu.return
    }
    func.func private @my_device_func(i32) -> i32 attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}
    func.func private @my_device_sub(i32, memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_1]>}
  }
}

// CHECK-LABEL: gpu.func @test
// CHECK: func.call @__wrapper_my_device_func
// CHECK: func.call @__wrapper_my_device_sub
// CHECK-LABEL: gpu.func @test2
// CHECK: func.call @__wrapper_my_device_func
// CHECK: func.call @__wrapper_my_device_sub
// CHECK: func.func private @__wrapper_my_device_func
// CHECK: func.func private @__wrapper_my_device_sub

// -----

// A routine and its callee may exist only in the GPU module.
module {
  gpu.module @local_only {
    acc.routine @r_local func(@wrapped_local) seq bind("actual_local")
    func.func private @wrapped_local()
        attributes {acc.routine_info = #acc.routine_info<[@r_local]>}
    gpu.func @caller() {
      func.call @wrapped_local() : () -> ()
      gpu.return
    }
  }
}

// CHECK-LABEL: gpu.module @local_only
// CHECK-LABEL: gpu.func @caller
// CHECK: func.call @actual_local() : () -> ()
// CHECK: func.func private @actual_local()
