// RUN: tr-opt %s --tr-autotune-reduction='rows=1024 cols=1024' | FileCheck %s

// Milestone 26: async / software pipelining is evaluated only after the
// baseline is measured. Row-sum intensity is ~1/4 flop/byte; extra smem
// and registers for double buffering do not pay off. The tuner keeps
// asyncDepth = 0.

// CHECK-DAG: tr.tune.async_helps_row_sum = false
// CHECK-DAG: tr.tune.async_depth = 0
// CHECK-DAG: tr.cost.baseline_row_sum_us
// CHECK-DAG: tr.cost.async_row_sum_us
// CHECK-DAG: tr.schedule.use_shared_memory = false

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
