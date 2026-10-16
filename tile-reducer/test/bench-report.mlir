// RUN: tr-opt %s --tr-bench-report='rows=4096 cols=4096' | FileCheck %s

// Milestone 27: benchmark record for a broader shape. Fields are
// analytical estimates until a device run is wired (Milestone 28+).

// CHECK-LABEL: func.func @row_sum
// CHECK-DAG: tr.bench.latency_us
// CHECK-DAG: tr.bench.effective_gbs
// CHECK-DAG: tr.bench.threads_per_block = 256
// CHECK-DAG: tr.bench.registers_per_thread
// CHECK-DAG: tr.bench.shared_memory_bytes = 0
// CHECK-DAG: tr.bench.occupancy
// CHECK-DAG: tr.bench.kernel_count = 1

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
