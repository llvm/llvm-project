// RUN: mlir-opt -split-input-file -test-xegpu-coalesce-gather-scatter %s | FileCheck %s
// RUN: mlir-opt -split-input-file -test-xegpu-coalesce-gather-scatter="max-chunk-size=4" %s | FileCheck --check-prefix=CHECK4 %s

// -----
// vector.step offsets -> stride 1, fully coalescible.
// 32 lanes, subgroup_size = 16 (default) -> lane_layout = 16, perLane = 2,
// max-chunk-size = 8 -> bound = min(32, 8, 2) = 2 -> lane_data = 2.
// The trivial `chunk_size = 1` attribute is dropped on success since the
// new lane_data FCD > 1 supersedes it.
// CHECK-LABEL: func.func @load_step_offsets(
// CHECK:   %[[STEP:.*]] = vector.step : vector<32xindex>
// CHECK:   %[[LOAD:.*]] = xegpu.load
// CHECK-NOT: chunk_size
// CHECK-SAME: layout = #xegpu.layout<inst_data = [32], lane_layout = [16], lane_data = [2]>
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
// CHECK:   return %[[LOAD]]
func.func @load_step_offsets(%ptr: i64) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// max-chunk-size = 4: same lane_layout / lane_data because the bound
// `min(contiguity = 32, budget = 4, perLane = 2) = 2` already saturates at
// perLane.
// CHECK4-LABEL: func.func @load_step_offsets_chunk4(
// CHECK4: xegpu.load
// CHECK4-SAME: layout = #xegpu.layout<inst_data = [32], lane_layout = [16], lane_data = [2]>
// CHECK4-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
func.func @load_step_offsets_chunk4(%ptr: i64) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// Dense constant arithmetic progression with stride 1.
// CHECK-LABEL: func.func @load_dense_ap_offsets(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<inst_data = [32], lane_layout = [16], lane_data = [2]>
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xi32>
func.func @load_dense_ap_offsets(%ptr: i64) -> vector<32xi32> {
  %offsets = arith.constant dense<[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
  ]> : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xi32>
  return %v : vector<32xi32>
}

// -----
// Non-stride-1 (stride 4) offsets: not contiguous, no layout attached.
// CHECK-LABEL: func.func @load_stride4_unchanged(
// CHECK: xegpu.load
// CHECK-NOT: lane_data
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
func.func @load_stride4_unchanged(%ptr: i64) -> vector<32xf32> {
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
// Non-uniform mask: not coalesced.
// CHECK-LABEL: func.func @load_partial_mask_unchanged(
// CHECK: xegpu.load
// CHECK-NOT: lane_data
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
func.func @load_partial_mask_unchanged(%ptr: i64, %mask: vector<32xi1>) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// Store with vector.step offsets coalesces.
// CHECK-LABEL: func.func @store_step_offsets(
// CHECK: xegpu.store
// CHECK-SAME: layout = #xegpu.layout<inst_data = [32], lane_layout = [16], lane_data = [2]>
// CHECK-SAME: : vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
func.func @store_step_offsets(%ptr: i64, %v: vector<32xf32>) {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  xegpu.store %v, %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
  return
}

// -----
// Store with all-equal offsets is left alone (ambiguous semantics).
// CHECK-LABEL: func.func @store_broadcast_offsets_unchanged(
// CHECK: xegpu.store
// CHECK-NOT: lane_data
// CHECK-SAME: vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
func.func @store_broadcast_offsets_unchanged(%ptr: i64, %v: vector<32xf32>) {
  %offsets = arith.constant dense<0> : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  xegpu.store %v, %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
  return
}

// -----
// memref-source variant of load coalesces too.
// CHECK-LABEL: func.func @load_memref_step(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<inst_data = [32], lane_layout = [16], lane_data = [2]>
// CHECK-SAME: : memref<1024xf32>, vector<32xindex>, vector<32xi1> -> vector<32xf32>
func.func @load_memref_step(%m: memref<1024xf32>) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %m[%offsets], %mask <{chunk_size = 1 : i64}>
      : memref<1024xf32>, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// 2-D offsets with leading unit dim: inner dim treated as lane dim.
// vector<1x32xindex> stride-1, subgroup_size = 16 ->
// lane_layout = [1, 16], lane_data = [1, 2].
// CHECK-LABEL: func.func @load_2d_leading_unit(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<inst_data = [1, 32], lane_layout = [1, 16], lane_data = [1, 2]>
// CHECK-SAME: : i64, vector<1x32xindex>, vector<1x32xi1> -> vector<1x32xf32>
func.func @load_2d_leading_unit(%ptr: i64) -> vector<1x32xf32> {
  %step = vector.step : vector<32xindex>
  %offsets = vector.shape_cast %step : vector<32xindex> to vector<1x32xindex>
  %mask = arith.constant dense<true> : vector<1x32xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<1x32xindex>, vector<1x32xi1> -> vector<1x32xf32>
  return %v : vector<1x32xf32>
}

// -----
// True 2-D dense AP: each row stride-1, inner = 16 = subgroup_size, so
// perLane = 1 and there's no room for lane_data > 1. The pass leaves the
// op alone; the wider variant `load_2d_dense_ap_wide` below exercises the
// coalescing path.
// CHECK-LABEL: func.func @load_2d_dense_ap(
// CHECK: xegpu.load
// CHECK-NOT: lane_data
// CHECK-SAME: : i64, vector<2x16xindex>, vector<2x16xi1> -> vector<2x16xf32>
func.func @load_2d_dense_ap(%ptr: i64) -> vector<2x16xf32> {
  %offsets = arith.constant dense<[
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
  ]> : vector<2x16xindex>
  %mask = arith.constant dense<true> : vector<2x16xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<2x16xindex>, vector<2x16xi1> -> vector<2x16xf32>
  return %v : vector<2x16xf32>
}

// -----
// True 2-D dense AP with inner > subgroup_size: each row stride-1 across 32
// lanes. With subgroup_size = 16 (from #xevm.target chip = "pvc"),
// lane_layout[inner] = 16, perLane = 32 / 16 = 2, contiguity[inner] = 32,
// budget = max-chunk-size / 1 = 8. bound = min(32, 8, 2) = 2 -> lane_data = 2.
// CHECK-LABEL: gpu.func @load_2d_dense_ap_wide(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<inst_data = [1, 32], lane_layout = [1, 16], lane_data = [1, 2]>
// CHECK-SAME: : i64, vector<2x32xindex>, vector<2x32xi1> -> vector<2x32xf32>
gpu.module @kernel [#xevm.target<chip = "pvc">] {
  gpu.func @load_2d_dense_ap_wide(%ptr: i64) -> vector<2x32xf32> {
    %offsets = arith.constant dense<[
      [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
      [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
       48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
    ]> : vector<2x32xindex>
    %mask = arith.constant dense<true> : vector<2x32xi1>
    %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
        : i64, vector<2x32xindex>, vector<2x32xi1> -> vector<2x32xf32>
    gpu.return %v : vector<2x32xf32>
  }
}

// -----
// 1-D step reshape_cast'ed to 2x32: inner extent 32 > subgroup_size = 16, so
// the lane_layout-first rule picks lane_layout[inner] = 16 (perLane = 2),
// then takes lane_data[inner] = min(contiguity = 32, budget = 8, perLane = 2)
// rounded down to a power-of-2 divisor of perLane => 2.
// CHECK-LABEL: gpu.func @load_2x32_step_shape_cast(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<inst_data = [1, 32], lane_layout = [1, 16], lane_data = [1, 2]>
// CHECK-SAME: : i64, vector<2x32xindex>, vector<2x32xi1> -> vector<2x32xf32>
gpu.module @kernel_2x32 [#xevm.target<chip = "pvc">] {
  gpu.func @load_2x32_step_shape_cast(%ptr: i64) -> vector<2x32xf32> {
    %step = vector.step : vector<64xindex>
    %offsets = vector.shape_cast %step : vector<64xindex> to vector<2x32xindex>
    %mask = arith.constant dense<true> : vector<2x32xi1>
    %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
        : i64, vector<2x32xindex>, vector<2x32xi1> -> vector<2x32xf32>
    gpu.return %v : vector<2x32xf32>
  }
}

// -----
// Negative: 2-D dense values where inner row is not stride-1 AP. Should
// not coalesce.
// CHECK-LABEL: func.func @load_2d_non_ap_unchanged(
// CHECK: xegpu.load
// CHECK-NOT: lane_data
// CHECK-SAME: : i64, vector<2x16xindex>, vector<2x16xi1> -> vector<2x16xf32>
func.func @load_2d_non_ap_unchanged(%ptr: i64) -> vector<2x16xf32> {
  %offsets = arith.constant dense<[
    [0, 1, 2, 3, 4, 5, 6, 7, 100, 9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
  ]> : vector<2x16xindex>
  %mask = arith.constant dense<true> : vector<2x16xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<2x16xindex>, vector<2x16xi1> -> vector<2x16xf32>
  return %v : vector<2x16xf32>
}

// -----
// "Reduction kernel" pattern with inner > subgroup_size: 2-D offsets built
// from `transpose(broadcast(rowOffsets))` + `broadcast(step)`. The transpose
// supplies inner-dim constancy, the broadcast(step) supplies inner-dim
// contiguity, and the addi recovers contiguity through visitAddSub. With
// inner = 32 and subgroup_size = 16 this picks lane_layout = [1, 16],
// lane_data = [1, 2].
// CHECK-LABEL: gpu.func @load_reduction_pattern_wide(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<inst_data = [1, 32], lane_layout = [1, 16], lane_data = [1, 2]>
// CHECK-SAME: : i64, vector<2x32xindex>, vector<2x32xi1> -> vector<2x32xf32>
gpu.module @kernel_reduction [#xevm.target<chip = "pvc">] {
  gpu.func @load_reduction_pattern_wide(%ptr: i64, %row0: index, %row1: index)
      -> vector<2x32xf32> {
    // Per-row base offsets.
    %r0   = vector.broadcast %row0 : index to vector<1xindex>
    %r1   = vector.broadcast %row1 : index to vector<1xindex>
    %rows = vector.shuffle %r0, %r1 [0, 1] : vector<1xindex>, vector<1xindex>
    // Inner stride-1 step.
    %step = vector.step : vector<32xindex>
    // Build 2x32 offsets: broadcast rows -> transpose -> add broadcast(step).
    %rowsBc = vector.broadcast %rows
            : vector<2xindex> to vector<32x2xindex>
    %rowsT  = vector.transpose %rowsBc, [1, 0]
            : vector<32x2xindex> to vector<2x32xindex>
    %cols2  = vector.broadcast %step
            : vector<32xindex> to vector<2x32xindex>
    %off    = arith.addi %rowsT, %cols2 : vector<2x32xindex>
    %mask   = arith.constant dense<true> : vector<2x32xi1>
    %v = xegpu.load %ptr[%off], %mask <{chunk_size = 1 : i64}>
        : i64, vector<2x32xindex>, vector<2x32xi1> -> vector<2x32xf32>
    gpu.return %v : vector<2x32xf32>
  }
}

// -----
// `divui` by a uniform constant equal to the inner stride recovers stride-1
// contiguity. Here offsets = [0,2,4,…,62] / 2 = [0,1,…,31], 32 lanes
// against subgroup_size = 16 -> lane_layout = 16, perLane = 2, contiguity
// = 32, bound = min(32, 8, 2) = 2 -> lane_data = 2.
// CHECK-LABEL: gpu.func @load_divui_recovers_contiguity(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<inst_data = [32], lane_layout = [16], lane_data = [2]>
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
gpu.module @kernel_divui [#xevm.target<chip = "pvc">] {
  gpu.func @load_divui_recovers_contiguity(%ptr: i64) -> vector<32xf32> {
    %even = arith.constant dense<[
       0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
      32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62]>
        : vector<32xindex>
    %c2   = arith.constant dense<2> : vector<32xindex>
    %offsets = arith.divui %even, %c2 : vector<32xindex>
    %mask = arith.constant dense<true> : vector<32xi1>
    %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
        : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
    gpu.return %v : vector<32xf32>
  }
}

// -----
// `divsi` by a uniform constant equal to the inner stride: same recovery as
// the divui case.
// CHECK-LABEL: gpu.func @load_divsi_recovers_contiguity(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<inst_data = [32], lane_layout = [16], lane_data = [2]>
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
gpu.module @kernel_divsi [#xevm.target<chip = "pvc">] {
  gpu.func @load_divsi_recovers_contiguity(%ptr: i64) -> vector<32xf32> {
    %even = arith.constant dense<[
       0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
      32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62]>
        : vector<32xindex>
    %c2   = arith.constant dense<2> : vector<32xindex>
    %offsets = arith.divsi %even, %c2 : vector<32xindex>
    %mask = arith.constant dense<true> : vector<32xi1>
    %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
        : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
    gpu.return %v : vector<32xf32>
  }
}

// -----
// `remui` by a constant that divides the inner stride: every element of a
// row is the same residue class -> inner-dim constant. The decision picks
// the broadcast case, which is currently disabled, so the load is left
// alone (no layout, no chunk_size change).
// CHECK-LABEL: gpu.func @load_remui_inner_uniform(
// CHECK: xegpu.load
// CHECK-NOT: lane_data
// CHECK-SAME: : i64, vector<16xindex>, vector<16xi1> -> vector<16xf32>
gpu.module @kernel_remui [#xevm.target<chip = "pvc">] {
  gpu.func @load_remui_inner_uniform(%ptr: i64) -> vector<16xf32> {
    %even = arith.constant dense<[
      0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]>
        : vector<16xindex>
    %c2   = arith.constant dense<2> : vector<16xindex>
    // (even % 2) is uniformly 0 along the inner dim.
    %offsets = arith.remui %even, %c2 : vector<16xindex>
    %mask = arith.constant dense<true> : vector<16xi1>
    %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
        : i64, vector<16xindex>, vector<16xi1> -> vector<16xf32>
    gpu.return %v : vector<16xf32>
  }
}

// -----
// `andi` with a power-of-two-minus-one mask is `% (1 << k)`. With stride 2
// and mask 1 (= 2-1), every result element is uniform: same broadcast case
// as remui above, currently disabled, so the load is left alone.
// CHECK-LABEL: gpu.func @load_andi_inner_uniform(
// CHECK: xegpu.load
// CHECK-NOT: lane_data
// CHECK-SAME: : i64, vector<16xindex>, vector<16xi1> -> vector<16xf32>
gpu.module @kernel_andi [#xevm.target<chip = "pvc">] {
  gpu.func @load_andi_inner_uniform(%ptr: i64) -> vector<16xf32> {
    %even = arith.constant dense<[
      0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]>
        : vector<16xindex>
    %m = arith.constant dense<1> : vector<16xindex>
    %offsets = arith.andi %even, %m : vector<16xindex>
    %mask = arith.constant dense<true> : vector<16xi1>
    %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
        : i64, vector<16xindex>, vector<16xi1> -> vector<16xf32>
    gpu.return %v : vector<16xf32>
  }
}

// -----
// `shli` by a constant scales the inner stride. step << 1 -> stride 2,
// which on its own is not coalescible (no divui follows), so we get no
// layout attached even when there's room (vector<32>, perLane=2).
// CHECK-LABEL: gpu.func @load_shli_unchanged(
// CHECK: xegpu.load
// CHECK-NOT: lane_data
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
gpu.module @kernel_shli [#xevm.target<chip = "pvc">] {
  gpu.func @load_shli_unchanged(%ptr: i64) -> vector<32xf32> {
    %step = vector.step : vector<32xindex>
    %k = arith.constant dense<1> : vector<32xindex>
    %offsets = arith.shli %step, %k : vector<32xindex>
    %mask = arith.constant dense<true> : vector<32xi1>
    %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
        : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
    gpu.return %v : vector<32xf32>
  }
}

// -----
// `shli` followed by `shrui` cancels: `(step << 1) >> 1` has innerStride
// scaled to 2 then divided by 2 -> stride 1 again. With inner = 32 and
// subgroup_size = 16 we coalesce by lane_data = 2.
// CHECK-LABEL: gpu.func @load_shli_then_shrui_recovers(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<inst_data = [32], lane_layout = [16], lane_data = [2]>
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
gpu.module @kernel_shli_shrui [#xevm.target<chip = "pvc">] {
  gpu.func @load_shli_then_shrui_recovers(%ptr: i64) -> vector<32xf32> {
    %step = vector.step : vector<32xindex>
    %k = arith.constant dense<1> : vector<32xindex>
    %doubled = arith.shli %step, %k : vector<32xindex>
    %offsets = arith.shrui %doubled, %k : vector<32xindex>
    %mask = arith.constant dense<true> : vector<32xi1>
    %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
        : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
    gpu.return %v : vector<32xf32>
  }
}

// -----
// `arith.select` between two AP arms with the same inner-dim properties:
// the result inherits the meet of the two arms. Both arms here are
// stride-1 step + a per-arm constant base, so the select preserves
// inner-dim contiguity = 32 -> coalesces with lane_data = 2.
// CHECK-LABEL: gpu.func @load_select_two_aps(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<inst_data = [32], lane_layout = [16], lane_data = [2]>
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
gpu.module @kernel_select [#xevm.target<chip = "pvc">] {
  gpu.func @load_select_two_aps(%ptr: i64, %cond: i1) -> vector<32xf32> {
    %step = vector.step : vector<32xindex>
    %a = arith.constant dense<0>  : vector<32xindex>
    %b = arith.constant dense<64> : vector<32xindex>
    %baseA = arith.addi %step, %a : vector<32xindex>
    %baseB = arith.addi %step, %b : vector<32xindex>
    %offsets = arith.select %cond, %baseA, %baseB : vector<32xindex>
    %mask = arith.constant dense<true> : vector<32xi1>
    %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
        : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
    gpu.return %v : vector<32xf32>
  }
}

// -----
// `divui` by a constant that does NOT divide the inner stride: stride 2 / 3
// is not exact, so we conservatively give up. No layout attached.
// CHECK-LABEL: gpu.func @load_divui_non_divisor_unchanged(
// CHECK: xegpu.load
// CHECK-NOT: lane_data
// CHECK-SAME: : i64, vector<16xindex>, vector<16xi1> -> vector<16xf32>
gpu.module @kernel_divui_neg [#xevm.target<chip = "pvc">] {
  gpu.func @load_divui_non_divisor_unchanged(%ptr: i64) -> vector<16xf32> {
    %even = arith.constant dense<[
      0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]>
        : vector<16xindex>
    %c3   = arith.constant dense<3> : vector<16xindex>
    %offsets = arith.divui %even, %c3 : vector<16xindex>
    %mask = arith.constant dense<true> : vector<16xi1>
    %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
        : i64, vector<16xindex>, vector<16xi1> -> vector<16xf32>
    gpu.return %v : vector<16xf32>
  }
}

// -----
// 2-D store with step + shape_cast offsets coalesces too.
// CHECK-LABEL: func.func @store_2d_step(
// CHECK: xegpu.store
// CHECK-SAME: layout = #xegpu.layout<inst_data = [1, 32], lane_layout = [1, 16], lane_data = [1, 2]>
// CHECK-SAME: : vector<1x32xf32>, i64, vector<1x32xindex>, vector<1x32xi1>
func.func @store_2d_step(%ptr: i64, %v: vector<1x32xf32>) {
  %step = vector.step : vector<32xindex>
  %offsets = vector.shape_cast %step : vector<32xindex> to vector<1x32xindex>
  %mask = arith.constant dense<true> : vector<1x32xi1>
  xegpu.store %v, %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : vector<1x32xf32>, i64, vector<1x32xindex>, vector<1x32xi1>
  return
}

// -----
// An op that already declares chunk_size = 2 is left alone: the verifier
// requires a particular value/mask shape relationship for chunked ops, so
// the pass conservatively skips when an explicit chunk_size > 1 is set
// (a downstream pass has already committed to that per-lane chunked
// access).
// CHECK-LABEL: gpu.func @load_explicit_chunk_unchanged(
// CHECK: xegpu.load
// CHECK-NOT: lane_data
// CHECK-SAME: <{chunk_size = 2 : i64}>
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32x2xf32>
gpu.module @kernel_explicit_chunk [#xevm.target<chip = "pvc">] {
  gpu.func @load_explicit_chunk_unchanged(%ptr: i64) -> vector<32x2xf32> {
    %offsets = vector.step : vector<32xindex>
    %mask = arith.constant dense<true> : vector<32xi1>
    %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 2 : i64}>
        : i64, vector<32xindex>, vector<32xi1> -> vector<32x2xf32>
    gpu.return %v : vector<32x2xf32>
  }
}
