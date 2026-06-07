// RUN: mlir-opt -xevm-attach-target='chip=pvc' \
// RUN:   -test-xegpu-propagate-layouts="layout-kind=lane" -split-input-file %s \
// RUN:   | FileCheck %s

// Lane-level layout propagation runs the coalesce-gather-scatter analysis up
// front. A coalescing store seeds a non-trivial lane_data on its FCD, which
// flows backward through the def-use chain to producers. A consumer that
// needs a different lane_data (e.g. a reduction over the FCD) overrides the
// hint, so no unlowerable xegpu.convert_layout is created.

// -----
// No-conflict path: load -> elementwise -> store, all contiguous on a
// vector<32> (inner 32 > subgroup_size 16). The coalescing store seeds
// lane_data = [2]; it flows back through the mulf to the load, so the whole
// chain is laid out with lane_layout = [16], lane_data = [2].
// CHECK-LABEL: gpu.func @coalesce_load_ew_store(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<lane_layout = [16], lane_data = [2]>
// CHECK: arith.mulf
// CHECK-SAME: layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [2]>
// CHECK: xegpu.store
// CHECK-SAME: layout = #xegpu.layout<lane_layout = [16], lane_data = [2]>
// CHECK-NOT: xegpu.convert_layout
gpu.module @kernel_chain {
  gpu.func @coalesce_load_ew_store(%src: i64, %dst: i64) {
    %step = vector.step : vector<32xindex>
    %mask = arith.constant dense<true> : vector<32xi1>
    %v = xegpu.load %src[%step], %mask <{chunk_size = 1 : i64}>
        : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
    %c = arith.constant dense<2.0> : vector<32xf32>
    %p = arith.mulf %v, %c : vector<32xf32>
    xegpu.store %p, %dst[%step], %mask <{chunk_size = 1 : i64}>
        : vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
    gpu.return
  }
}

// -----
// Coalesced FCD-reduction path: the load's offsets are contiguous on a
// vector<2x32> (inner 32 > subgroup_size 16). Its consumer is an
// associative-commutative multi_reduction over the FCD (inner dim). The load
// is coalesced (lane_data = [1, 2]: each lane owns 2 contiguous elements), and
// the reduction's SOURCE-operand layout adopts that coalesced FCD value (so it
// agrees with the load — no convert). The reduction RESULT lives on the
// post-reduction layout with lane_data = [1, 1] (the factor only applies to
// the source the lane folds locally), so result/accumulator match the real
// consumer. The downstream SgToLane reduction folds the 2 per-lane elements
// before the cross-lane butterfly.
// CHECK-LABEL: gpu.func @reduction_coalesces(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 2]>
// CHECK: vector.multi_reduction
// CHECK-SAME: #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>
// CHECK-NOT: xegpu.convert_layout
// CHECK-NOT: xegpu.convert_layout
gpu.module @kernel_reduction {
  gpu.func @reduction_coalesces(%src: i64, %dst: i64, %acc: vector<2xf32>) {
    %step = vector.step : vector<64xindex>
    %off2d = vector.shape_cast %step : vector<64xindex> to vector<2x32xindex>
    %mask = arith.constant dense<true> : vector<2x32xi1>
    %v = xegpu.load %src[%off2d], %mask <{chunk_size = 1 : i64}>
        : i64, vector<2x32xindex>, vector<2x32xi1> -> vector<2x32xf32>
    %r = vector.multi_reduction <add>, %v, %acc [1]
        : vector<2x32xf32> to vector<2xf32>
    %roff = vector.step : vector<2xindex>
    %rmask = arith.constant dense<true> : vector<2xi1>
    xegpu.store %r, %dst[%roff], %rmask <{chunk_size = 1 : i64}>
        : vector<2xf32>, i64, vector<2xindex>, vector<2xi1>
    gpu.return
  }
}

// -----
// The coalesce hint is a transient analysis artifact: it must not survive
// into the propagator's output.
// CHECK-LABEL: gpu.func @no_leftover_hint(
// CHECK-NOT: coalesce_hint
gpu.module @kernel_cleanup {
  gpu.func @no_leftover_hint(%src: i64, %dst: i64) {
    %step = vector.step : vector<32xindex>
    %mask = arith.constant dense<true> : vector<32xi1>
    %v = xegpu.load %src[%step], %mask <{chunk_size = 1 : i64}>
        : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
    xegpu.store %v, %dst[%step], %mask <{chunk_size = 1 : i64}>
        : vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
    gpu.return
  }
}
