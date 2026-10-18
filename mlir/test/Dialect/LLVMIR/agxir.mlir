// RUN: mlir-opt %s -split-input-file -verify-diagnostics | mlir-opt | FileCheck %s

// CHECK-LABEL: @special_registers
func.func @special_registers() -> i32 {
  // CHECK: agxir.thread.position.in.grid.x : i32
  %0 = agxir.thread.position.in.grid.x : i32
  // CHECK: agxir.thread.position.in.threadgroup.y range <i32, 0, 1024> : i32
  %1 = agxir.thread.position.in.threadgroup.y range <i32, 0, 1024> : i32
  // CHECK: agxir.threadgroup.position.in.grid.z : i32
  %2 = agxir.threadgroup.position.in.grid.z : i32
  // CHECK: agxir.threads.per.threadgroup.x : i32
  %3 = agxir.threads.per.threadgroup.x : i32
  // CHECK: agxir.threadgroups.per.grid.y : i32
  %4 = agxir.threadgroups.per.grid.y : i32
  // CHECK: agxir.thread.index.in.simdgroup : i32
  %5 = agxir.thread.index.in.simdgroup : i32
  // CHECK: agxir.simdgroup.index.in.threadgroup : i32
  %6 = agxir.simdgroup.index.in.threadgroup : i32
  // CHECK: agxir.total.active.threads.in.simdgroup : i32
  %7 = agxir.total.active.threads.in.simdgroup : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @sync
func.func @sync() {
  // CHECK: agxir.threadgroup.barrier
  agxir.threadgroup.barrier
  // CHECK: agxir.memory.barrier 1, 2, 9
  agxir.memory.barrier 1, 2, 9
  // CHECK: agxir.wait 0
  agxir.wait 0
  llvm.return
}

// CHECK-LABEL: @simdgroup
func.func @simdgroup(%v: f32, %h: f16, %i: i32, %p: i1) -> f32 {
  // CHECK: agxir.simd.ballot %{{.*}} : i32
  %0 = agxir.simd.ballot %p : i32
  // CHECK: agxir.quad.ballot %{{.*}} : i32
  %1 = agxir.quad.ballot %p : i32
  // CHECK: agxir.simd.shuffle %{{.*}}, %{{.*}} : f32
  %2 = agxir.simd.shuffle %v, %i : f32
  // CHECK: agxir.simd.shuffle.xor %{{.*}}, %{{.*}} : f16
  %3 = agxir.simd.shuffle.xor %h, %i : f16
  // CHECK: agxir.quad.shuffle.down %{{.*}}, %{{.*}} : i32
  %4 = agxir.quad.shuffle.down %i, %i : i32
  // CHECK: agxir.simd.reduce fadd %{{.*}} : f32
  %5 = agxir.simd.reduce fadd %v : f32
  // CHECK: agxir.quad.prefix iadd %{{.*}} : i32
  %6 = agxir.quad.prefix iadd %i : i32
  // CHECK: agxir.simd.prefix umax %{{.*}} : i32
  %7 = agxir.simd.prefix umax %i : i32
  // CHECK: agxir.rcp %{{.*}} : f32
  %8 = agxir.rcp %v : f32
  // CHECK: agxir.rsqrt %{{.*}} : f16
  %9 = agxir.rsqrt %h : f16
  // CHECK: agxir.dfdx %{{.*}} : f32
  %10 = agxir.dfdx %v : f32
  llvm.return %2 : f32
}

// CHECK: gpu.module @targets [#agxir.target, #agxir.target<O = 3, chip = "agx-g13c">]
gpu.module @targets [#agxir.target, #agxir.target<O = 3, chip = "agx-g13c">] {
}

// -----

func.func @bad_reduce(%v: f32) -> f32 {
  // expected-error @below {{reduction 'iadd' does not apply to 'f32'}}
  %0 = agxir.simd.reduce iadd %v : f32
  llvm.return %0 : f32
}
