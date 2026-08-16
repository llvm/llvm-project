// RUN: tr-opt %s --convert-tr-row-sum-to-gpu | FileCheck %s

// Milestone 17: 128x128 logical tile, 256 threads, 8 warps.
//   warp w -> rows w, w+8, ..., 120     (16 sequential rows, not 128 warps)
//   lane L -> columns L, L+32, L+64, L+96
// Register-reduce the four lane values, then gpu.subgroup_reduce.
// tr.program_id stays logical; it is not threadIdx.

// CHECK-LABEL: func.func @row_sum
// CHECK-SAME: memref<?x?xf32>
// CHECK-SAME: memref<?xf32>
// CHECK-SAME: gpu.known_block_size = array<i32: 256, 1, 1>
func.func @row_sum(%in: !tr.buffer<MxKxf32>, %out: !tr.buffer<Mxf32>) {
  %row_blk     = tr.program_id 0 : index
  %c128        = arith.constant 128 : index
  %k           = tr.dim %in, 1 : !tr.buffer<MxKxf32>, index
  %num_k_tiles = arith.divui %k, %c128 : index
  %zero = tr.constant 0.0 : !tr.tile<128xf32>
  %result = tr.for %kt = 0 to %num_k_tiles step 1
      iter_args(%acc = %zero) -> !tr.tile<128xf32> {
    %t       = tr.load %in[%row_blk, %kt]
        : !tr.buffer<MxKxf32>, !tr.tile<128x128xf32>
    %partial = tr.reduce_sum %t, axis = 1
        : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
    %acc2    = tr.add %acc, %partial : !tr.tile<128xf32>
    tr.yield %acc2 : !tr.tile<128xf32>
  }
  tr.store %out[%row_blk], %result : !tr.buffer<Mxf32>, !tr.tile<128xf32>
  return
}

// CHECK: %[[PID:.*]] = tr.program_id 0 : index
// CHECK: gpu.thread_id x
// CHECK: %[[LANE:.*]] = gpu.lane_id
// CHECK: %[[WARP:.*]] = gpu.subgroup_id
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[BASE:.*]] = arith.muli %[[PID]], %[[C128]]

// 16 sequential rows per warp: localRow = warp + s * 8
// CHECK: scf.for %[[S:.*]] = %{{.*}} to %[[C16]]
// CHECK:   %[[S8:.*]] = arith.muli %[[S]], %[[C8]]
// CHECK:   %[[LOCAL:.*]] = arith.addi %[[WARP]], %[[S8]]
// CHECK:   arith.addi %[[BASE]], %[[LOCAL]]

// 4 elements per lane: col = kt * 128 + lane + j * 32
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[C4]]
// CHECK:   arith.muli %{{.*}}, %[[C32]]
// CHECK:   arith.addi %{{.*}}, %[[LANE]]

// CHECK: gpu.subgroup_reduce add %{{.*}}
// CHECK: memref.store

// Column reduction is not this map; leave it alone.
// CHECK-LABEL: func.func @column
// CHECK: tr.reduce_sum
func.func @column(%t: !tr.tile<128x128xf32>) -> !tr.tile<128xf32> {
  %r = tr.reduce_sum %t, axis = 0 : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
  return %r : !tr.tile<128xf32>
}
