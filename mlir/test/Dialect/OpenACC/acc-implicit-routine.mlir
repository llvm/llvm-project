// RUN: mlir-opt %s -acc-implicit-routine -split-input-file | FileCheck %s

// -----

// Implicit acc routine for a function called from a compute region.
module {
  func.func @callee() {
    return
  }
  func.func @test_serial_call() {
    acc.serial {
      func.call @callee() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: acc.routine @acc_routine_0 func(@callee) implicit
// CHECK: func.func @callee() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}
// CHECK: func.func @test_serial_call

// -----

// Call from a compute region to a function that calls another function:
// both callees get implicit acc routines (nested via routine walk).
module {
  func.func @leaf() {
    return
  }
  func.func @middle() {
    func.call @leaf() : () -> ()
    return
  }
  func.func @test_nested_calls() {
    acc.serial {
      func.call @middle() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: acc.routine @acc_routine_1 func(@leaf) implicit
// CHECK: func.func @leaf() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_1]>}
// CHECK: acc.routine @acc_routine_0 func(@middle) implicit
// CHECK: func.func @middle() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}
// CHECK: func.func @test_nested_calls

// -----

// Function called from acc.loop nested in a compute region.
module {
  func.func @loop_callee() {
    return
  }
  func.func @test_loop_call() {
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    acc.parallel {
      acc.loop control(%i : index) = (%c1 : index) to (%c10 : index) step (%c1 : index) {
        func.call @loop_callee() : () -> ()
        acc.yield
      } attributes {independent = [#acc.device_type<none>]}
      acc.yield
    }
    return
  }
}

// CHECK: acc.routine @acc_routine_0 func(@loop_callee) implicit
// CHECK: func.func @loop_callee() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}
// CHECK: func.func @test_loop_call

// -----

// Implicit routine for a call inside acc.kernels.
module {
  func.func @kernels_callee() {
    return
  }
  func.func @test_kernels_call() {
    acc.kernels {
      func.call @kernels_callee() : () -> ()
      acc.terminator
    }
    return
  }
}

// CHECK: acc.routine @acc_routine_0 func(@kernels_callee) implicit
// CHECK: func.func @kernels_callee() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}
// CHECK: func.func @test_kernels_call

// -----

// Cyclic calls between routines: both get implicit routines without looping.
module {
  func.func @cycle_a() {
    func.call @cycle_b() : () -> ()
    return
  }
  func.func @cycle_b() {
    func.call @cycle_a() : () -> ()
    return
  }
  func.func @test_cyclic_calls() {
    acc.serial {
      func.call @cycle_a() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: acc.routine @acc_routine_0 func(@cycle_a) implicit
// CHECK: func.func @cycle_a() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}
// CHECK: acc.routine @acc_routine_1 func(@cycle_b) implicit
// CHECK: func.func @cycle_b() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_1]>}
// CHECK: func.func @test_cyclic_calls

// -----

// Callee that already has an explicit acc routine does not get a second one.
module {
  acc.routine @r_explicit func(@host_fn) seq
  func.func @host_fn() attributes {acc.routine_info = #acc.routine_info<[@r_explicit]>} {
    return
  }
  func.func @test_explicit_routine_callee() {
    acc.serial {
      func.call @host_fn() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: acc.routine @r_explicit func(@host_fn) seq
// CHECK-NOT: acc.routine @acc_routine_0 func(@host_fn)

// -----

// acc.routine with bind: recursive walk is skipped for calls only in that routine.
module {
  acc.routine @r_bind func(@foo) seq bind(@bar)
  func.func @foo() attributes {acc.routine_info = #acc.routine_info<[@r_bind]>} {
    func.call @only_in_bound_routine() : () -> ()
    return
  }
  func.func @bar() {
    return
  }
  func.func @only_in_bound_routine() {
    return
  }
  func.func @test_bind_skips_recursive() {
    acc.serial {
      func.call @foo() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: acc.routine @r_bind func(@foo) bind(@bar) seq
// CHECK-NOT: acc.routine @{{.*}} func(@only_in_bound_routine)

// -----

// LLVM intrinsic declaration called from a compute region: no implicit routine.
module {
  func.func private @llvm.sqrt.f32(%a: f32) -> f32
  func.func @test_llvm_intrinsic(%x: f32) {
    acc.serial {
      %0 = func.call @llvm.sqrt.f32(%x) : (f32) -> f32
      acc.yield
    }
    return
  }
}

// CHECK: func.call @llvm.sqrt.f32
// CHECK-NOT: acc.routine @{{.*}} func(@llvm.sqrt.f32)

// -----

// Call outside a compute region does not create an implicit routine.
module {
  func.func @outside_callee() {
    return
  }
  func.func @test_call_outside_region() {
    func.call @outside_callee() : () -> ()
    acc.serial {
      acc.yield
    }
    return
  }
}

// CHECK: func.func @test_call_outside_region
// CHECK: call @outside_callee
// CHECK-NOT: acc.routine @{{.*}} func(@outside_callee)

// -----

// Indirect call in a compute region: callee is an SSA value, not a symbol.
module {
  func.func @test_indirect_in_serial(%callee: () -> ()) {
    acc.serial {
      func.call_indirect %callee() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: func.func @test_indirect_in_serial
// CHECK: func.call_indirect %{{.*}}() : () -> ()
// CHECK-NOT: acc.routine @acc_routine_

// -----

// Indirect call inside an existing acc routine.
module {
  acc.routine @explicit_routine func(@routine_caller) seq
  func.func @routine_caller(%callee: () -> ()) attributes {acc.routine_info = #acc.routine_info<[@explicit_routine]>} {
    func.call_indirect %callee() : () -> ()
    return
  }
}

// CHECK: acc.routine @explicit_routine func(@routine_caller) seq
// CHECK: func.func @routine_caller
// CHECK: call_indirect %{{.*}}() : () -> ()
// CHECK-NOT: acc.routine @acc_routine_

// -----

// Call to a symbol with no definition in the module must not crash and must
// not create an implicit acc routine.
module {
  func.func @test_undefined_callee() {
    %buf = memref.alloca() : memref<f32>
    acc.serial {
      test.call_and_store @undefined_callee(%buf), %buf {store_before_call = false} : (memref<f32>, memref<f32>) -> ()
      acc.yield
    }
    return
  }
}

// CHECK: func.func @test_undefined_callee
// CHECK: test.call_and_store @undefined_callee
// CHECK-NOT: acc.routine @acc_routine_
// CHECK-NOT: acc.routine_info
