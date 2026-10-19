// RUN: mlir-opt %s -convert-gpu-to-agxir -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-gpu-to-agxir='index-bitwidth=32' -split-input-file | FileCheck --check-prefix=CHECK32 %s

gpu.module @test_module {
  // CHECK-LABEL: llvm.func @indices
  // CHECK32-LABEL: llvm.func @indices
  func.func @indices() -> (index, index, index, index, index, index, index) {
    // CHECK: agxir.thread.position.in.threadgroup.x : i32
    // CHECK: llvm.sext %{{.*}} : i32 to i64
    // CHECK32: agxir.thread.position.in.threadgroup.x : i32
    // CHECK32-NOT: llvm.sext
    %tid = gpu.thread_id x
    // CHECK: agxir.threadgroup.position.in.grid.y
    %bid = gpu.block_id y
    // CHECK: agxir.threads.per.threadgroup.z
    %bdim = gpu.block_dim z
    // CHECK: agxir.threadgroups.per.grid.x
    %gdim = gpu.grid_dim x
    // CHECK: agxir.threadgroup.position.in.grid.x
    // CHECK: agxir.threads.per.threadgroup.x
    // CHECK: llvm.mul
    // CHECK: agxir.thread.position.in.threadgroup.x
    // CHECK: llvm.add
    %gid = gpu.global_id x
    // CHECK: agxir.thread.index.in.simdgroup
    %lane = gpu.lane_id
    // CHECK: llvm.mlir.constant(32 : i64) : i64
    %size = gpu.subgroup_size : index
    func.return %tid, %bid, %bdim, %gdim, %gid, %lane, %size
        : index, index, index, index, index, index, index
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: llvm.func @barrier
  func.func @barrier() {
    // CHECK: llvm.fence syncscope("workgroup") release
    // CHECK-NEXT: agxir.threadgroup.barrier
    // CHECK-NEXT: llvm.fence syncscope("workgroup") acquire
    gpu.barrier
    func.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: llvm.func @shuffle
  func.func @shuffle(%v: f32, %o: i32, %w: i32) -> (f32, i1) {
    // CHECK: %[[LANE:.*]] = agxir.thread.index.in.simdgroup : i32
    // CHECK: %[[SRC:.*]] = llvm.xor %[[LANE]], %{{.*}} : i32
    // CHECK: %[[VALID:.*]] = llvm.icmp "slt" %[[SRC]], %{{.*}} : i32
    // CHECK: %[[SEL:.*]] = llvm.select %[[VALID]], %[[SRC]], %[[LANE]] : i1, i32
    // CHECK: agxir.simd.shuffle %{{.*}}, %[[SEL]] : i32
    %r, %valid = gpu.shuffle xor %v, %o, %w : f32
    func.return %r, %valid : f32, i1
  }

  // CHECK-LABEL: llvm.func @ballot_reduce
  func.func @ballot_reduce(%p: i1, %x: f32) -> (i32, f32) {
    // CHECK: agxir.simd.ballot %{{.*}} : i32
    %b = gpu.ballot %p : i32
    // CHECK: agxir.simd.reduce fadd %{{.*}} : f32
    %r = gpu.subgroup_reduce add %x : (f32) -> f32
    func.return %b, %r : i32, f32
  }
}

// -----

gpu.module @test_module {
  // Kernels use the agx_kernel calling convention; workgroup attributions
  // are threadgroup memory.
  // CHECK: llvm.mlir.global internal @__wg_kernel_0() {addr_space = 3 : i32} : !llvm.array<32 x f32>
  // CHECK-LABEL: llvm.func agx_kernelcc @kernel
  // CHECK-SAME: agxir.kernel
  gpu.func @kernel(%arg0: memref<32xf32, #gpu.address_space<global>>)
      workgroup(%tile: memref<32xf32, #gpu.address_space<workgroup>>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %arg0[%c0] : memref<32xf32, #gpu.address_space<global>>
    memref.store %v, %tile[%c0] : memref<32xf32, #gpu.address_space<workgroup>>
    gpu.return
  }
}
