// RUN: mlir-opt -xevm-attach-target='chip=bmg' \
// RUN:   -xegpu-propagate-layout="layout-kind=inst" -xegpu-blocking \
// RUN:   -xegpu-propagate-layout="layout-kind=lane" -xegpu-sg-to-lane-distribute \
// RUN:   -canonicalize -cse -convert-xegpu-to-xevm -canonicalize -cse \
// RUN:   -split-input-file %s | FileCheck %s

// From contiguity to the final memory accesses. Each lane's access must stay
// inside one contiguous run of the mem_desc, because the lowering reads or
// writes the whole per-lane vector at the address of its first element.

// -----
// 16-wide blocks, 96-wide tile: 6 elements per lane would cross a block edge,
// so every lane writes 2.
// CHECK-LABEL: gpu.func @store_matrix_blocked
// CHECK-NOT: vector<{{[3-9]}}xf32>
// CHECK-COUNT-24: llvm.store %{{.*}}, %{{.*}} : vector<2xf32>, !llvm.ptr<3>
// CHECK-NOT: llvm.store
// CHECK: gpu.return
gpu.module @test {
gpu.func @store_matrix_blocked(
    %dst: !xegpu.mem_desc<8x96xf32, #xegpu.mem_layout<block = [8, 16]>>) {
  %v = arith.constant dense<1.0> : vector<8x96xf32>
  xegpu.store_matrix %v, %dst[0, 0]
      : vector<8x96xf32>, !xegpu.mem_desc<8x96xf32, #xegpu.mem_layout<block = [8, 16]>>
  gpu.return
}
}

// -----
// 3-wide blocks: runs of 3 hold no chunk of 2, so the load reads one element
// per lane. The 2-wide store then gets its pairs through a lane shuffle.
// CHECK-LABEL: gpu.func @load_matrix_blocked
// CHECK-NOT: llvm.load %{{.*}} : !llvm.ptr<3> -> vector
// CHECK-COUNT-48: llvm.load %{{.*}} : !llvm.ptr<3> -> f32
// CHECK: xevm.bitcast_shuffle
// CHECK: llvm.store %{{.*}}, %{{.*}} : vector<2xf32>, !llvm.ptr<3>
gpu.module @test {
gpu.func @load_matrix_blocked(
    %src: !xegpu.mem_desc<8x96xf32, #xegpu.mem_layout<block = [8, 3]>>,
    %dst: !xegpu.mem_desc<8x96xf32>) {
  %v = xegpu.load_matrix %src[0, 0]
      : !xegpu.mem_desc<8x96xf32, #xegpu.mem_layout<block = [8, 3]>> -> vector<8x96xf32>
  %e = math.exp %v : vector<8x96xf32>
  xegpu.store_matrix %e, %dst[0, 0] : vector<8x96xf32>, !xegpu.mem_desc<8x96xf32>
  gpu.return
}
}

// -----
// An inst_data-only anchor is completed with a chunk that tiles the runs:
// 16-wide blocks and inst_data = [1, 32] give 2 elements per lane.
// CHECK-LABEL: gpu.func @store_matrix_partial_anchor_blocked_16
// CHECK-NOT: vector<{{[3-9]}}xf32>
// CHECK-COUNT-24: llvm.store %{{.*}}, %{{.*}} : vector<2xf32>, !llvm.ptr<3>
// CHECK-NOT: llvm.store
// CHECK: gpu.return
gpu.module @test {
gpu.func @store_matrix_partial_anchor_blocked_16(
    %dst: !xegpu.mem_desc<8x96xf32, #xegpu.mem_layout<block = [8, 16]>>) {
  %v = arith.constant dense<1.0> : vector<8x96xf32>
  xegpu.store_matrix %v, %dst[0, 0] <{layout = #xegpu.layout<inst_data = [1, 32]>}>
      : vector<8x96xf32>, !xegpu.mem_desc<8x96xf32, #xegpu.mem_layout<block = [8, 16]>>
  gpu.return
}
}

// -----
// 3-wide blocks allow one element per lane, and inst_data = [1, 16] fits that
// exactly: every access is a single element.
// CHECK-LABEL: gpu.func @store_matrix_partial_anchor_blocked_3
// CHECK-NOT: llvm.store %{{.*}}, %{{.*}} : vector
// CHECK-COUNT-48: llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr<3>
// CHECK-NOT: llvm.store
// CHECK: gpu.return
gpu.module @test {
gpu.func @store_matrix_partial_anchor_blocked_3(
    %dst: !xegpu.mem_desc<8x96xf32, #xegpu.mem_layout<block = [8, 3]>>) {
  %v = arith.constant dense<1.0> : vector<8x96xf32>
  xegpu.store_matrix %v, %dst[0, 0] <{layout = #xegpu.layout<inst_data = [1, 16]>}>
      : vector<8x96xf32>, !xegpu.mem_desc<8x96xf32, #xegpu.mem_layout<block = [8, 3]>>
  gpu.return
}
}
