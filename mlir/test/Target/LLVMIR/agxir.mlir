// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: define agx_kernel void @kernel(ptr addrspace(1) %0)
llvm.func @kernel(%out: !llvm.ptr<1>) attributes {agxir.kernel} {
  // CHECK: call i32 @llvm.agx.thread.position.in.grid.x()
  %0 = agxir.thread.position.in.grid.x : i32
  // CHECK: call range(i32 0, 64) i32 @llvm.agx.thread.position.in.threadgroup.y()
  %1 = agxir.thread.position.in.threadgroup.y range <i32, 0, 64> : i32
  // CHECK: call i32 @llvm.agx.threadgroups.per.grid.z()
  %2 = agxir.threadgroups.per.grid.z : i32
  // CHECK: call i32 @llvm.agx.thread.index.in.simdgroup()
  %3 = agxir.thread.index.in.simdgroup : i32
  // CHECK: call void @llvm.agx.threadgroup.barrier()
  agxir.threadgroup.barrier
  // CHECK: call void @llvm.agx.memory.barrier(i32 1, i32 2, i32 9)
  agxir.memory.barrier 1, 2, 9
  %f = llvm.bitcast %0 : i32 to f32
  // CHECK: call float @llvm.agx.simd.shuffle.xor.f32(float %{{.*}}, i32 %{{.*}})
  %4 = agxir.simd.shuffle.xor %f, %3 : f32
  // CHECK: call float @llvm.agx.simd.reduce.f32(float %{{.*}}, i32 1)
  %5 = agxir.simd.reduce fadd %4 : f32
  // CHECK: call i32 @llvm.agx.quad.prefix.i32(i32 %{{.*}}, i32 30)
  %6 = agxir.quad.prefix umax %1 : i32
  %p = llvm.icmp "ugt" %6, %2 : i32
  // CHECK: call i32 @llvm.agx.simd.ballot(i1 %{{.*}})
  %7 = agxir.simd.ballot %p : i32
  // CHECK: call float @llvm.agx.rsqrt.f32(float %{{.*}})
  %8 = agxir.rsqrt %5 : f32
  llvm.store %8, %out : f32, !llvm.ptr<1>
  llvm.return
}

// Functions without the attribute keep their calling convention.
// CHECK: define void @helper()
llvm.func @helper() {
  llvm.return
}

// CHECK-LABEL: define void @extras
llvm.func @extras(%out: !llvm.ptr<1>, %v: vector<2xf32>, %m: i32, %x: f32) {
  // CHECK: call i32 @llvm.agx.core.id()
  %0 = agxir.core.id : i32
  // CHECK: call i32 @llvm.agx.active.thread.index.in.quadgroup()
  %1 = agxir.active.thread.index.in.quadgroup : i32
  // CHECK: call i32 @llvm.agx.total.active.threads.in.quadgroup()
  %2 = agxir.total.active.threads.in.quadgroup : i32
  // CHECK: call void @llvm.agx.set.fpmode(i32 %{{.*}})
  agxir.set.fpmode %m
  // CHECK: call i32 @llvm.agx.get.fpmode()
  %3 = agxir.get.fpmode : i32
  // CHECK: call <2 x float> @llvm.agx.simdgroup.matrix.fma.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
  %4 = agxir.simdgroup.matrix.fma %v, %v, %v : vector<2xf32>
  // CHECK: call float @llvm.agx.rsqrt.special.f32(float %{{.*}})
  %5 = agxir.rsqrt.special %x : f32
  llvm.store %4, %out : vector<2xf32>, !llvm.ptr<1>
  llvm.return
}
