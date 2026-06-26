// RUN: mlir-opt -split-input-file -test-xegpu-coalesce-gather-scatter %s | FileCheck %s
// RUN: mlir-opt -split-input-file -test-xegpu-coalesce-gather-scatter="max-chunk-size=4" %s | FileCheck --check-prefix=CHECK4 %s

// Driver test: turns the `contiguity` attribute the analysis stamps into a
// lane_layout / lane_data layout. The contiguity analysis itself is covered by
// contiguity-analysis.mlir; here we only check that the driver derives the
// layout from a stamped (or already-contiguous) access. The subgroup size
// comes from the target (pvc -> 16), so each case carries an xevm.target.

// -----
// Contiguous 32-element load, subgroup_size = 16: lane_layout = 16,
// perLane = 2, lane_data = min(contiguity, max-chunk-size, perLane) = 2.
// CHECK-LABEL: gpu.func @load_step(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<inst_data = [32], lane_layout = [16], lane_data = [2]>
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
gpu.module @kernel [#xevm.target<chip = "pvc">] {
  gpu.func @load_step(%ptr: i64) -> vector<32xf32> {
    %offsets = vector.step : vector<32xindex>
    %mask = arith.constant dense<true> : vector<32xi1>
    %v = xegpu.load %ptr[%offsets], %mask
        : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
    gpu.return %v : vector<32xf32>
  }
}

// -----
// Store mirrors the load.
// CHECK-LABEL: gpu.func @store_step(
// CHECK: xegpu.store
// CHECK-SAME: layout = #xegpu.layout<inst_data = [32], lane_layout = [16], lane_data = [2]>
// CHECK-SAME: : vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
gpu.module @kernel_store [#xevm.target<chip = "pvc">] {
  gpu.func @store_step(%ptr: i64, %v: vector<32xf32>) {
    %offsets = vector.step : vector<32xindex>
    %mask = arith.constant dense<true> : vector<32xi1>
    xegpu.store %v, %ptr[%offsets], %mask
        : vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
    gpu.return
  }
}

// -----
// Non-contiguous (stride-4) offsets: the analysis stamps nothing, so the
// driver leaves the op unchanged.
// CHECK-LABEL: gpu.func @load_stride4_unchanged(
// CHECK: xegpu.load
// CHECK-NOT: lane_data
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
gpu.module @kernel_stride4 [#xevm.target<chip = "pvc">] {
  gpu.func @load_stride4_unchanged(%ptr: i64) -> vector<32xf32> {
    %c4 = arith.constant 4 : index
    %step = vector.step : vector<32xindex>
    %splat = vector.broadcast %c4 : index to vector<32xindex>
    %offsets = arith.muli %step, %splat : vector<32xindex>
    %mask = arith.constant dense<true> : vector<32xi1>
    %v = xegpu.load %ptr[%offsets], %mask
        : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
    gpu.return %v : vector<32xf32>
  }
}

// -----
// 2-D leading-unit offsets: lane_layout / lane_data are placed on the inner
// dim -> [1, 16] / [1, 2].
// CHECK-LABEL: gpu.func @load_2d_leading_unit(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<inst_data = [1, 32], lane_layout = [1, 16], lane_data = [1, 2]>
// CHECK-SAME: : i64, vector<1x32xindex>, vector<1x32xi1> -> vector<1x32xf32>
gpu.module @kernel_2d [#xevm.target<chip = "pvc">] {
  gpu.func @load_2d_leading_unit(%ptr: i64) -> vector<1x32xf32> {
    %step = vector.step : vector<32xindex>
    %offsets = vector.shape_cast %step : vector<32xindex> to vector<1x32xindex>
    %mask = arith.constant dense<true> : vector<1x32xi1>
    %v = xegpu.load %ptr[%offsets], %mask
        : i64, vector<1x32xindex>, vector<1x32xi1> -> vector<1x32xf32>
    gpu.return %v : vector<1x32xf32>
  }
}

// -----
// max-chunk-size = 4 saturates at perLane = 2 anyway, so lane_data stays 2.
// CHECK4-LABEL: gpu.func @load_step_chunk4(
// CHECK4: xegpu.load
// CHECK4-SAME: layout = #xegpu.layout<inst_data = [32], lane_layout = [16], lane_data = [2]>
gpu.module @kernel_chunk4 [#xevm.target<chip = "pvc">] {
  gpu.func @load_step_chunk4(%ptr: i64) -> vector<32xf32> {
    %offsets = vector.step : vector<32xindex>
    %mask = arith.constant dense<true> : vector<32xi1>
    %v = xegpu.load %ptr[%offsets], %mask
        : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
    gpu.return %v : vector<32xf32>
  }
}

// -----
// No target: the driver cannot size the lanes, so it leaves the op unchanged
// (the stamped contiguity is just removed).
// CHECK-LABEL: func.func @load_no_target_unchanged(
// CHECK: xegpu.load
// CHECK-NOT: lane_data
// CHECK-NOT: contiguity
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
func.func @load_no_target_unchanged(%ptr: i64) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}
