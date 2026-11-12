// RUN: tr-opt %s --tr-autotune-reduction='rows=1 cols=100000000' | FileCheck %s --check-prefix=LARGEK
// RUN: tr-opt %s --tr-autotune-reduction='rows=16384 cols=4096' | FileCheck %s --check-prefix=BASE

// Milestone 25: bounded legal schedules, analytical prune, cache by
//   kind | axis | dtype | tile | shape bucket | arch | compiler
// Do not tune every exact shape.

// LARGEK-DAG: tr.tune.shape_bucket = "M_few_K_large"
// LARGEK-DAG: tr.tune.cache_key = "row|1|f32|128x128|M_few_K_large|a100-like|tile-reducer-23"
// LARGEK-DAG: tr.tune.k_splits = 64
// LARGEK-DAG: tr.tune.async_depth = 0
// LARGEK-DAG: tr.tune.async_helps_row_sum = false

// BASE-DAG: tr.tune.shape_bucket = "M_many_K_medium"
// BASE-DAG: tr.tune.k_splits = 1
// BASE-DAG: tr.tune.async_depth = 0

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
