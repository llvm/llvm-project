// RUN: mlir-opt -split-input-file \
// RUN:   -test-xegpu-coalesce-gather-scatter="analyze-only=true" %s | FileCheck %s

// Analyze-only mode: stamps the `contiguous_chunk` attribute on contiguous ops
// and leaves the layout unchanged. This test pins the attribute contract that
// the coalescing consumer (and downstream integrations) read.

// -----
// 1-D vector.step, fully contiguous -> contiguous_chunk = 32 (the inner extent).
// CHECK-LABEL: func.func @load_step_offsets(
// CHECK: xegpu.load
// CHECK-SAME: <{contiguous_chunk = 32 : i64}>
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
// CHECK-NOT: lane_data
func.func @load_step_offsets(%ptr: i64) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// 2-D leading-1 dim: contiguity computed on the inner dim -> contiguous_chunk = 32.
// CHECK-LABEL: func.func @load_2d_leading_unit(
// CHECK: xegpu.load
// CHECK-SAME: <{contiguous_chunk = 32 : i64}>
// CHECK-SAME: : i64, vector<1x32xindex>, vector<1x32xi1> -> vector<1x32xf32>
// CHECK-NOT: lane_data
func.func @load_2d_leading_unit(%ptr: i64) -> vector<1x32xf32> {
  %step = vector.step : vector<32xindex>
  %offsets = vector.shape_cast %step : vector<32xindex> to vector<1x32xindex>
  %mask = arith.constant dense<true> : vector<1x32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<1x32xindex>, vector<1x32xi1> -> vector<1x32xf32>
  return %v : vector<1x32xf32>
}

// -----
// Stride-4 offsets: not contiguous, no attribute stamped.
// CHECK-LABEL: func.func @load_stride4_no_chunk(
// CHECK: xegpu.load
// CHECK-NOT: contiguous_chunk
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
func.func @load_stride4_no_chunk(%ptr: i64) -> vector<32xf32> {
  %c4 = arith.constant 4 : index
  %step = vector.step : vector<32xindex>
  %splat = vector.broadcast %c4 : index to vector<32xindex>
  %offsets = arith.muli %step, %splat : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// All-equal offsets: inner dim is constant (not contiguous), so no attribute
// is stamped.
// CHECK-LABEL: func.func @load_broadcast_offsets_no_chunk(
// CHECK: xegpu.load
// CHECK-NOT: contiguous_chunk
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
func.func @load_broadcast_offsets_no_chunk(%ptr: i64) -> vector<32xf32> {
  %offsets = arith.constant dense<0> : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// Store with vector.step offsets: contiguous_chunk = 32 stamped on the store.
// CHECK-LABEL: func.func @store_step_chunk(
// CHECK: xegpu.store
// CHECK-SAME: <{contiguous_chunk = 32 : i64}>
// CHECK-SAME: : vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
// CHECK-NOT: lane_data
func.func @store_step_chunk(%ptr: i64, %v: vector<32xf32>) {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  xegpu.store %v, %ptr[%offsets], %mask
      : vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
  return
}

// -----
// A pre-existing contiguous_chunk takes precedence: the analysis must not
// overwrite it.
// CHECK-LABEL: func.func @user_chunk_preserved(
// CHECK: xegpu.load
// CHECK-SAME: <{contiguous_chunk = 2 : i64}>
func.func @user_chunk_preserved(%ptr: i64) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{contiguous_chunk = 2 : i64}>
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}
