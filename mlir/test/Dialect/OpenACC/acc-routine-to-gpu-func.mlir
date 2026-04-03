// RUN: mlir-opt %s -acc-routine-to-gpu-func -split-input-file | FileCheck %s

// CHECK: gpu.module @acc_gpu_module {
// CHECK: gpu.func @host_foo
// CHECK: memref.store
// CHECK: gpu.return
acc.routine @routine_seq func(@host_foo) seq
func.func @host_foo(%buf: memref<8xi32>) attributes {acc.specialized_routine = #acc.specialized_routine<@routine_seq, <seq>, "host_foo">} {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  memref.store %c0_i32, %buf[%c0] : memref<8xi32>
  return
}

// -----

// Bind routine is erased; materialized routine is moved to GPU module.
// CHECK: gpu.module @acc_gpu_module {
// CHECK: gpu.func @host_foo
// CHECK: gpu.return
// CHECK-NOT: acc.routine @routine_bind
acc.routine @routine_seq func(@host_foo) seq
acc.routine @routine_bind func(@host_bind) seq bind("myname")
func.func @host_foo() attributes {acc.specialized_routine = #acc.specialized_routine<@routine_seq, <seq>, "host_foo">} {
  return
}
func.func @host_bind() {
  return
}

// -----

// One routine with body, one declaration-only; both end up in GPU module.
// CHECK: acc.routine @acc_routine_0
// CHECK: acc.routine @acc_routine_1
// CHECK: gpu.module @{{.*}}
// CHECK-NEXT: gpu.func @devicefunc(){{.*}} {
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: func.func private @declfunc()
module {
  acc.routine @acc_routine_0 func(@devicefunc)
  func.func @devicefunc() attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_0, <seq>, "devicefunc">} {
    return
  }
  acc.routine @acc_routine_1 func(@declfunc)
  func.func private @declfunc() -> ()
}

// -----

// nohost routine: host copy is removed after moving to GPU module.
// CHECK: acc.routine @acc_routine_0
// CHECK-NOT: func.func @nohost_vec
module {
  acc.routine @acc_routine_0 func(@nohost_vec) vector nohost
  func.func @nohost_vec(%arg0: i32) attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_0, <vector>, "nohost_vec">} {
    return
  }
}
