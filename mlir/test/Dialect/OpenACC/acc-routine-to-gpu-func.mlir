// RUN: mlir-opt %s -acc-routine-to-gpu-func -split-input-file | FileCheck %s
// RUN: mlir-opt %s -acc-routine-to-gpu-func=device-type=nvidia -split-input-file | FileCheck %s --check-prefix=DEVICE-NVIDIA

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

// Routine does not name device_type(nvidia) so the bind("myname") under device_type(*)
// applies and the pass does not clone `star_bound` into the GPU module
// DEVICE-NVIDIA-LABEL: acc.routine @star_bind func(@star_bound)
// DEVICE-NVIDIA: func.func @star_bound()
// DEVICE-NVIDIA: gpu.module @acc_gpu_module {
// DEVICE-NVIDIA-NEXT: gpu.func @host_foo_star
// DEVICE-NVIDIA: gpu.return
// DEVICE-NVIDIA-NOT: gpu.func @star_bound
acc.routine @routine_seq_star func(@host_foo_star) seq
acc.routine @star_bind func(@star_bound) seq bind("myname" [#acc.device_type<star>])
func.func @host_foo_star() attributes {acc.specialized_routine = #acc.specialized_routine<@routine_seq_star, <seq>, "host_foo_star">} {
  return
}
func.func @star_bound() {
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
