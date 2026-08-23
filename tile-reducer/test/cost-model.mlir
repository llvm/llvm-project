// RUN: tr-opt %s --tr-estimate-reduction-cost='rows=1024 cols=1024' | FileCheck %s

// Milestone 23: baseline row-sum schedule and roofline cost.
// T ~= max(T_compute, T_memory) + T_sync + T_launch + T_tail.
// Not cycle-exact. Row-sum baseline uses no shared memory.

// CHECK-LABEL: func.func @row_sum
// CHECK-DAG: tr.schedule.threads_per_block = 256
// CHECK-DAG: tr.schedule.warps_per_block = 8
// CHECK-DAG: tr.schedule.rows_per_warp = 16
// CHECK-DAG: tr.schedule.elements_per_lane = 4
// CHECK-DAG: tr.schedule.use_shared_memory = false
// CHECK-DAG: tr.schedule.async_depth = 0
// CHECK-DAG: tr.schedule.k_splits = 1
// CHECK-DAG: tr.cost.legal = true
// CHECK-DAG: tr.cost.t_total
// CHECK-DAG: tr.cost.occupancy

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
