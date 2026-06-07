// RUN: mlir-opt -split-input-file \
// RUN:   -test-xegpu-coalesce-gather-scatter="analyze-only=true" %s | FileCheck %s

// Analyze-only mode: stamps `xegpu.coalesce_hint` on coalescible ops and
// leaves the layout / chunk_size unchanged. This test pins the hint
// attribute contract that the apply API (and downstream propagator
// integrations) consume.

// -----
// 1-D vector.step, fully coalescible -> hint with factor = 2 stamped.
// CHECK-LABEL: func.func @load_step_offsets(
// CHECK: xegpu.load
// CHECK-SAME: <{chunk_size = 1 : i64}>
// CHECK-SAME: {xegpu.coalesce_hint = #xegpu.coalesce_hint<factor = 2 : i64>}
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
// CHECK-NOT: lane_data
func.func @load_step_offsets(%ptr: i64) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// 2-D leading-1 dim: hint stamped on the load with factor = 2.
// CHECK-LABEL: func.func @load_2d_leading_unit(
// CHECK: xegpu.load
// CHECK-SAME: {xegpu.coalesce_hint = #xegpu.coalesce_hint<factor = 2 : i64>}
// CHECK-SAME: : i64, vector<1x32xindex>, vector<1x32xi1> -> vector<1x32xf32>
// CHECK-NOT: lane_data
func.func @load_2d_leading_unit(%ptr: i64) -> vector<1x32xf32> {
  %step = vector.step : vector<32xindex>
  %offsets = vector.shape_cast %step : vector<32xindex> to vector<1x32xindex>
  %mask = arith.constant dense<true> : vector<1x32xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<1x32xindex>, vector<1x32xi1> -> vector<1x32xf32>
  return %v : vector<1x32xf32>
}

// -----
// Stride-4 offsets: not coalescible, no hint stamped.
// CHECK-LABEL: func.func @load_stride4_no_hint(
// CHECK: xegpu.load
// CHECK-NOT: coalesce_hint
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
func.func @load_stride4_no_hint(%ptr: i64) -> vector<32xf32> {
  %c4 = arith.constant 4 : index
  %step = vector.step : vector<32xindex>
  %splat = vector.broadcast %c4 : index to vector<32xindex>
  %offsets = arith.muli %step, %splat : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// All-equal offsets: classified as Broadcast by decide(); analysis returns
// None at the stamping stage, so no hint is stamped (broadcast-load
// rewrite was removed).
// CHECK-LABEL: func.func @load_broadcast_offsets_no_hint(
// CHECK: xegpu.load
// CHECK-NOT: coalesce_hint
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
func.func @load_broadcast_offsets_no_hint(%ptr: i64) -> vector<32xf32> {
  %offsets = arith.constant dense<0> : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// Explicit chunk_size > 1: precondition fails, no hint stamped (regardless
// of analysis result).
// CHECK-LABEL: gpu.func @load_explicit_chunk_no_hint(
// CHECK: xegpu.load
// CHECK-NOT: coalesce_hint
// CHECK-SAME: <{chunk_size = 2 : i64}>
gpu.module @kernel_explicit_chunk [#xevm.target<chip = "pvc">] {
  gpu.func @load_explicit_chunk_no_hint(%ptr: i64) -> vector<32x2xf32> {
    %offsets = vector.step : vector<32xindex>
    %mask = arith.constant dense<true> : vector<32xi1>
    %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 2 : i64}>
        : i64, vector<32xindex>, vector<32xi1> -> vector<32x2xf32>
    gpu.return %v : vector<32x2xf32>
  }
}

// -----
// Store with vector.step offsets: hint stamped with factor = 2 on the store
// op.
// CHECK-LABEL: func.func @store_step_hint(
// CHECK: xegpu.store
// CHECK-SAME: {xegpu.coalesce_hint = #xegpu.coalesce_hint<factor = 2 : i64>}
// CHECK-SAME: : vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
// CHECK-NOT: lane_data
func.func @store_step_hint(%ptr: i64, %v: vector<32xf32>) {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  xegpu.store %v, %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
  return
}
