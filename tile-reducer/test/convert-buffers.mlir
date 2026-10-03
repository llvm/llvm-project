// RUN: tr-opt %s --convert-tr-buffers-to-memref | FileCheck %s

// CHECK-LABEL: func.func @row_sum
// CHECK-SAME: (%[[IN:.*]]: memref<?x?xf32>, %[[OUT:.*]]: memref<?xf32>)
func.func @row_sum(%in: !tr.buffer<MxKxf32>, %out: !tr.buffer<Mxf32>) {
  %row_blk     = tr.program_id 0 : index
  %c128        = arith.constant 128 : index
  // CHECK: memref.dim %[[IN]], %{{.*}} : memref<?x?xf32>
  %k           = tr.dim %in, 1 : !tr.buffer<MxKxf32>, index
  %num_k_tiles = arith.divui %k, %c128 : index
  %zero = tr.constant 0.0 : !tr.tile<128xf32>

  %result = tr.for %kt = 0 to %num_k_tiles step 1
      iter_args(%acc = %zero) -> !tr.tile<128xf32> {
    // CHECK: tr.load %[[IN]]
    // CHECK-SAME: memref<?x?xf32>, !tr.tile<128x128xf32>
    %t = tr.load %in[%row_blk, %kt] : !tr.buffer<MxKxf32>, !tr.tile<128x128xf32>
    %partial = tr.reduce_sum %t, axis = 1 : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
    %acc2 = tr.add %acc, %partial : !tr.tile<128xf32>
    tr.yield %acc2 : !tr.tile<128xf32>
  }
  // CHECK: tr.store %[[OUT]]
  // CHECK-SAME: memref<?xf32>, !tr.tile<128xf32>
  tr.store %out[%row_blk], %result : !tr.buffer<Mxf32>, !tr.tile<128xf32>
  return
}
