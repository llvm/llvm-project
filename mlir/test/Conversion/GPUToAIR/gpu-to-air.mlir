// RUN: mlir-opt %s --convert-gpu-to-air=use-bare-ptr-memref-call-conv=1 --split-input-file | FileCheck %s

// System values are calls of AIR's functions, components of the
// three-dimensional ones; kernels carry "air-kernel"; memrefs are bare
// pointers in AIR's address spaces.
gpu.module @ids {
  // CHECK-LABEL: llvm.func @ids(
  // CHECK-SAME: %{{.*}}: !llvm.ptr<1>)
  // CHECK-SAME: passthrough = ["air-kernel"]
  gpu.func @ids(%out: memref<8xi32, 1>) kernel {
    // CHECK: %[[T:.*]] = llvm.call @air.thread_position_in_threadgroup.v3i32() : () -> vector<3xi32>
    // CHECK: llvm.extractelement %[[T]]
    %t = gpu.thread_id x
    // CHECK: llvm.call @air.threadgroup_position_in_grid.v3i32()
    %b = gpu.block_id y
    // gpu.global_id is block_id * block_dim + thread_id (populateGpuRewritePatterns).
    // CHECK: llvm.call @air.threadgroup_position_in_grid.v3i32()
    // CHECK: llvm.call @air.threads_per_threadgroup.v3i32()
    // CHECK: llvm.call @air.thread_position_in_threadgroup.v3i32()
    %g = gpu.global_id z
    // CHECK: llvm.call @air.thread_index_in_simdgroup() : () -> i32
    %l = gpu.lane_id
    // CHECK: llvm.call @air.simdgroup_index_in_threadgroup() : () -> i32
    %s = gpu.subgroup_id : index
    // CHECK: llvm.mlir.constant(32 : i64)
    %n = gpu.subgroup_size : index
    %x = arith.addi %t, %b : index
    %y = arith.addi %g, %l : index
    %z = arith.addi %s, %n : index
    %w = arith.addi %x, %y : index
    %v = arith.addi %w, %z : index
    %i = arith.index_cast %v : index to i32
    %c0 = arith.constant 0 : index
    memref.store %i, %out[%c0] : memref<8xi32, 1>
    gpu.return
  }
}

// -----

// The barrier orders device and threadgroup memory, or the memory its
// memfence names (none for `memfence []`); a subgroup-scope barrier is AIR's
// SIMD-group barrier. Shuffles, ballots and reductions are AIR's SIMD-group
// functions.
gpu.module @simd {
  // CHECK-LABEL: llvm.func @simd(
  gpu.func @simd(%out: memref<8xf32, 1>, %v: f32, %p: i1) kernel {
    // CHECK: %[[F:.*]] = llvm.mlir.constant(3 : i32)
    // CHECK: %[[S:.*]] = llvm.mlir.constant(1 : i32)
    // CHECK: llvm.call @air.wg.barrier(%[[F]], %[[S]]) : (i32, i32) -> ()
    gpu.barrier
    // CHECK: %[[N:.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: llvm.call @air.wg.barrier(%[[N]], %{{.*}}) : (i32, i32) -> ()
    gpu.barrier memfence []
    // CHECK: %[[W:.*]] = llvm.mlir.constant(2 : i32)
    // CHECK: llvm.call @air.wg.barrier(%[[W]], %{{.*}}) : (i32, i32) -> ()
    gpu.barrier memfence [#gpu.address_space<workgroup>]
    // CHECK: %[[G:.*]] = llvm.mlir.constant(1 : i32)
    // CHECK: llvm.call @air.simdgroup.barrier(%[[G]], %{{.*}}) : (i32, i32) -> ()
    gpu.barrier scope <subgroup> memfence [#gpu.address_space<global>]
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant 32 : i32
    // CHECK: %[[LANE:.*]] = llvm.call @air.thread_index_in_simdgroup()
    // CHECK: %[[NEGW:.*]] = llvm.sub %{{.*}}, %[[W:.*]] : i32
    // CHECK: %[[ADD:.*]] = llvm.add %[[LANE]], %[[W]] : i32
    // CHECK: %[[END:.*]] = llvm.and %[[ADD]], %[[NEGW]] : i32
    // CHECK: %[[SRC:.*]] = llvm.xor %[[LANE]], %{{.*}} : i32
    // CHECK: %[[START:.*]] = llvm.and %[[LANE]], %[[NEGW]] : i32
    // CHECK: %[[GE:.*]] = llvm.icmp "sge" %[[SRC]], %[[START]] : i32
    // CHECK: %[[LT:.*]] = llvm.icmp "slt" %[[SRC]], %[[END]] : i32
    // CHECK: %[[VALID:.*]] = llvm.and %[[GE]], %[[LT]] : i1
    // CHECK: llvm.select %[[VALID]], %[[SRC]], %[[LANE]] : i1, i32
    // CHECK: llvm.call @air.simd_shuffle.u.i32(%{{.*}}, %{{.*}}) : (i32, i16) -> i32
    %sh, %ok = gpu.shuffle xor %v, %c1, %c32 : f32
    // CHECK: llvm.call @air.simd_ballot.i64(%{{.*}}) : (i1) -> i64
    %m = gpu.ballot %p : i64
    // CHECK: llvm.call @air.simd_sum.f32(%{{.*}}) : (f32) -> f32
    %r = gpu.subgroup_reduce add %sh : (f32) -> f32
    %c0 = arith.constant 0 : index
    memref.store %r, %out[%c0] : memref<8xf32, 1>
    %mf = arith.uitofp %m : i64 to f32
    memref.store %mf, %out[%c0] : memref<8xf32, 1>
    gpu.return
  }
}

// -----

// SIMD-group matrices are AIR's: a pointer to the element at the indices, the
// strides from the memref (swapped for transpose), 8 x 8 dimensions.
gpu.module @matrix {
  // CHECK-LABEL: llvm.func @mm(
  gpu.func @mm(%a: memref<64xf32, 1>, %c: memref<64xf32, 1>) kernel {
    %i0 = arith.constant 0 : index
    %A = memref.reinterpret_cast %a to offset: [0], sizes: [8, 8], strides: [8, 1] : memref<64xf32, 1> to memref<8x8xf32, strided<[8, 1]>, 1>
    %C = memref.reinterpret_cast %c to offset: [0], sizes: [8, 8], strides: [8, 1] : memref<64xf32, 1> to memref<8x8xf32, strided<[8, 1]>, 1>
    // CHECK: llvm.call @air.simdgroup_matrix_8x8_load.v64f32.p1f32(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<1>, vector<2xi64>, vector<2xi64>, vector<2xi64>) -> vector<64xf32>
    %m = air.simdgroup_matrix_load %A[%i0, %i0] transpose : memref<8x8xf32, strided<[8, 1]>, 1> -> vector<64xf32>
    %one = arith.constant 1.0 : f32
    // CHECK: llvm.call @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(%{{.*}}) : (f32) -> vector<64xf32>
    %f = air.simdgroup_matrix_fill %one : vector<64xf32>
    // CHECK: llvm.call @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32
    %d = air.simdgroup_matrix_multiply_accumulate %m, %m, %f : vector<64xf32>
    // CHECK: llvm.call @air.simdgroup_matrix_8x8_store.v64f32.p1f32(
    air.simdgroup_matrix_store %d, %C[%i0, %i0] : vector<64xf32> -> memref<8x8xf32, strided<[8, 1]>, 1>
    %h = arith.truncf %m : vector<64xf32> to vector<64xf16>
    // CHECK: llvm.call @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<64xf16>, vector<64xf16>, vector<64xf32>) -> vector<64xf32>
    %e = air.simdgroup_matrix_multiply_accumulate %h, %h, %d : vector<64xf16>, vector<64xf32>
    air.simdgroup_matrix_store %e, %C[%i0, %i0] : vector<64xf32> -> memref<8x8xf32, strided<[8, 1]>, 1>
    gpu.return
  }
}

// -----

// AIR has no generic address space: a memref without a memory space is device
// memory, except an alloca, which is the thread's.
gpu.module @default_space {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-SAME: %{{.*}}: !llvm.ptr<1>
  gpu.func @k(%out: memref<4xf32>) kernel {
    %c0 = arith.constant 0 : index
    // CHECK: llvm.alloca %{{.*}} x f32 : (i64) -> !llvm.ptr
    %tmp = memref.alloca() : memref<4xf32>
    %v = memref.load %tmp[%c0] : memref<4xf32>
    // CHECK: llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr<1>
    memref.store %v, %out[%c0] : memref<4xf32>
    gpu.return
  }
}

// -----

// A known threadgroup size is an AGX hint on the kernel.
gpu.module @block_size {
  // CHECK-LABEL: llvm.func @k(
  // CHECK-SAME: passthrough = ["air-kernel", ["agx-max-threads-per-threadgroup", "128"]]
  gpu.func @k() kernel attributes {known_block_size = array<i32: 64, 2, 1>} {
    gpu.return
  }
}
