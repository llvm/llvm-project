// RUN: mlir-opt -split-input-file -xegpu-coalesce-gather-scatter %s | FileCheck %s
// RUN: mlir-opt -split-input-file -xegpu-coalesce-gather-scatter="max-chunk-size=4" %s | FileCheck --check-prefix=CHECK4 %s

// -----
// vector.step offsets -> stride 1, fully coalescible.
// 32 lanes, max-chunk-size = 8 -> factor 8 -> lane_layout=[4], lane_data=[8].
// CHECK-LABEL: func.func @load_step_offsets(
// CHECK:   %[[STEP:.*]] = vector.step : vector<32xindex>
// CHECK:   %[[LOAD:.*]] = xegpu.load
// CHECK-SAME: layout = #xegpu.layout<lane_layout = [4], lane_data = [8]>
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
// max-chunk-size = 4: factor 4 -> lane_layout=[8], lane_data=[4].
// CHECK4-LABEL: func.func @load_step_offsets_chunk4(
// CHECK4: xegpu.load
// CHECK4-SAME: layout = #xegpu.layout<lane_layout = [8], lane_data = [4]>
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
// CHECK-SAME: layout = #xegpu.layout<lane_layout = [4], lane_data = [8]>
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
// All-equal offsets -> broadcast load (single load + vector.broadcast).
// CHECK-LABEL: func.func @load_broadcast_offsets(
// CHECK: %[[L:.*]] = xegpu.load
// CHECK-SAME: : i64, vector<1xindex>, vector<1xi1> -> vector<1xf32>
// CHECK: %[[E:.*]] = vector.extract %[[L]][0]
// CHECK: %[[B:.*]] = vector.broadcast %[[E]] : f32 to vector<32xf32>
// CHECK: return %[[B]]
func.func @load_broadcast_offsets(%ptr: i64) -> vector<32xf32> {
  %offsets = arith.constant dense<0> : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
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
// CHECK-SAME: layout = #xegpu.layout<lane_layout = [4], lane_data = [8]>
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
// CHECK-SAME: layout = #xegpu.layout<lane_layout = [4], lane_data = [8]>
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
// vector<1x32xindex> stride-1 -> lane_data=[1, 8], lane_layout=[1, 4].
// CHECK-LABEL: func.func @load_2d_leading_unit(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<lane_layout = [1, 4], lane_data = [1, 8]>
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
// True 2-D dense AP: each row stride-1, 16 lanes per row -> lane_data=[1,8],
// lane_layout=[1, 2].
// CHECK-LABEL: func.func @load_2d_dense_ap(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<lane_layout = [1, 2], lane_data = [1, 8]>
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
// 1-D step reshape_cast'ed to 4x8: inner extent 8, factor 8
// -> lane_layout=[1, 1], lane_data=[1, 8] (each of 8 inner positions is a
// lane that fetches a contiguous chunk of 8).
// CHECK-LABEL: func.func @load_4x8_step_shape_cast(
// CHECK: xegpu.load
// CHECK-SAME: layout = #xegpu.layout<lane_layout = [1, 1], lane_data = [1, 8]>
// CHECK-SAME: : i64, vector<4x8xindex>, vector<4x8xi1> -> vector<4x8xf32>
func.func @load_4x8_step_shape_cast(%ptr: i64) -> vector<4x8xf32> {
  %step = vector.step : vector<32xindex>
  %offsets = vector.shape_cast %step : vector<32xindex> to vector<4x8xindex>
  %mask = arith.constant dense<true> : vector<4x8xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<4x8xindex>, vector<4x8xi1> -> vector<4x8xf32>
  return %v : vector<4x8xf32>
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
// 2-D store with step + shape_cast offsets coalesces too.
// CHECK-LABEL: func.func @store_2d_step(
// CHECK: xegpu.store
// CHECK-SAME: layout = #xegpu.layout<lane_layout = [1, 4], lane_data = [1, 8]>
// CHECK-SAME: : vector<1x32xf32>, i64, vector<1x32xindex>, vector<1x32xi1>
func.func @store_2d_step(%ptr: i64, %v: vector<1x32xf32>) {
  %step = vector.step : vector<32xindex>
  %offsets = vector.shape_cast %step : vector<32xindex> to vector<1x32xindex>
  %mask = arith.constant dense<true> : vector<1x32xi1>
  xegpu.store %v, %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : vector<1x32xf32>, i64, vector<1x32xindex>, vector<1x32xi1>
  return
}
