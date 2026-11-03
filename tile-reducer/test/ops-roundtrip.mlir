// RUN: tr-opt %s | tr-opt | FileCheck %s

// Canonical row-sum source. Milestone 3: every op parses and prints.

// CHECK-LABEL: func.func @row_sum
// CHECK: %[[PID:.*]] = tr.program_id 0 : index
// CHECK: %[[K:.*]] = tr.dim %{{.*}}, 1 : !tr.buffer<MxKxf32>, index
// CHECK: %[[ZERO:.*]] = tr.constant 0.000000e+00 : !tr.tile<128xf32>
// CHECK: %[[RES:.*]] = tr.for %{{.*}} = 0 to %{{.*}} step 1
// CHECK-SAME: iter_args(%{{.*}} = %[[ZERO]]) -> (!tr.tile<128xf32>)
// CHECK: %[[T:.*]] = tr.load %{{.*}}[%[[PID]], %{{.*}}] : !tr.buffer<MxKxf32>, !tr.tile<128x128xf32>
// CHECK: %[[P:.*]] = tr.reduce_sum %[[T]], axis = 1 : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
// CHECK: %[[A:.*]] = tr.add %{{.*}}, %[[P]] : !tr.tile<128xf32>
// CHECK: tr.yield %[[A]] : !tr.tile<128xf32>
// CHECK: tr.store %{{.*}}[%[[PID]]], %[[RES]] : !tr.buffer<Mxf32>, !tr.tile<128xf32>
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

    %acc2    = tr.add %acc, %partial
        : !tr.tile<128xf32>

    tr.yield %acc2 : !tr.tile<128xf32>
  }

  tr.store %out[%row_blk], %result : !tr.buffer<Mxf32>, !tr.tile<128xf32>
  return
}
