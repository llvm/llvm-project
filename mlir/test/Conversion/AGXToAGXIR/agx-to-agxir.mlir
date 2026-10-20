// RUN: mlir-opt %s -convert-agx-to-agxir | FileCheck %s

// CHECK-LABEL: @barrier
func.func @barrier() {
  // CHECK: llvm.fence syncscope("workgroup") release
  // CHECK-NEXT: agxir.threadgroup.barrier
  // CHECK-NEXT: llvm.fence syncscope("workgroup") acquire
  agx.threadgroup_barrier
  return
}

// CHECK-LABEL: @shuffle_scalar
func.func @shuffle_scalar(%f: f32, %h: f16, %i: i32) -> (f32, f16) {
  // CHECK: agxir.simd.shuffle.xor %{{.*}}, %{{.*}} : f32
  %0 = agx.shuffle xor %f, %i : f32
  // CHECK: agxir.quad.shuffle %{{.*}}, %{{.*}} : f16
  %1 = agx.shuffle idx %h, %i scope quad : f16
  return %0, %1 : f32, f16
}

// CHECK-LABEL: @shuffle_wide
func.func @shuffle_wide(%d: f64, %i: i32) -> f64 {
  // 64-bit values are shuffled in two 32-bit parts.
  // CHECK: agxir.simd.shuffle.down %{{.*}}, %{{.*}} : i32
  // CHECK: agxir.simd.shuffle.down %{{.*}}, %{{.*}} : i32
  // CHECK-NOT: agxir.simd.shuffle
  %0 = agx.shuffle down %d, %i : f64
  return %0 : f64
}

// CHECK-LABEL: @reduce
func.func @reduce(%f: f32, %i: i32, %s: i16, %h: f16) -> (f32, i32, i16, f16) {
  // CHECK: agxir.simd.reduce fadd %{{.*}} : f32
  %0 = agx.reduce add %f : f32
  // CHECK: agxir.quad.prefix umax %{{.*}} : i32
  %1 = agx.reduce maxui %i prefix scope quad : i32
  // 16-bit values are extended according to the signedness.
  // CHECK: llvm.sext %{{.*}} : i16 to i32
  // CHECK: agxir.simd.reduce smin %{{.*}} : i32
  // CHECK: llvm.trunc %{{.*}} : i32 to i16
  %2 = agx.reduce minsi %s : i16
  // CHECK: llvm.fpext %{{.*}} : f16 to f32
  // CHECK: agxir.simd.reduce fmax %{{.*}} : f32
  // CHECK: llvm.fptrunc %{{.*}} : f32 to f16
  %3 = agx.reduce maxnumf %h : f16
  return %0, %1, %2, %3 : f32, i32, i16, f16
}

// CHECK-LABEL: @ballot
func.func @ballot(%p: i1) -> (i32, i32) {
  // CHECK: agxir.simd.ballot %{{.*}} : i32
  %0 = agx.ballot %p : i32
  // CHECK: agxir.quad.ballot %{{.*}} : i32
  %1 = agx.ballot %p scope quad : i32
  return %0, %1 : i32, i32
}

// CHECK-LABEL: @memory_barriers
func.func @memory_barriers() {
  // CHECK: llvm.fence seq_cst
  agx.memory_barrier
  // CHECK: llvm.fence syncscope("workgroup") seq_cst
  agx.memory_barrier scope threadgroup
  return
}

// CHECK-LABEL: @mma
func.func @mma(%a: vector<2xf32>, %b: vector<2xf32>, %c: vector<2xf32>) -> vector<2xf32> {
  // CHECK: agxir.simdgroup.matrix.fma %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf32>
  %0 = agx.mma %a * %b + %c : vector<2xf32>, vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @fpmode
func.func @fpmode(%m: i32) -> i32 {
  // CHECK: agxir.set.fpmode %{{.*}}
  agx.set_fpmode %m : i32
  // CHECK: agxir.get.fpmode : i32
  %0 = agx.get_fpmode : i32
  return %0 : i32
}
