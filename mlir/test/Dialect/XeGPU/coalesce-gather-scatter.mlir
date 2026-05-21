// RUN: mlir-opt -split-input-file -xegpu-coalesce-gather-scatter %s | FileCheck %s
// RUN: mlir-opt -split-input-file -xegpu-coalesce-gather-scatter="max-chunk-size=4" %s | FileCheck --check-prefix=CHECK4 %s

// -----
// vector.step offsets -> stride 1, fully coalescible.
// 32 lanes, max-chunk-size = 8 -> factor 8 -> 4 lanes, chunk_size = 8.
// CHECK-LABEL: @load_step_offsets
// CHECK-SAME:    (%[[ARG:.*]]: i64) -> vector<32xf32>
// CHECK:   %[[STEP:.*]] = vector.step : vector<4xindex>
// CHECK:   %[[LOAD:.*]] = xegpu.load %[[ARG]][{{.*}}], {{.*}} <{chunk_size = 8 : i64}>
// CHECK-SAME:    : i64, vector<4xindex>, vector<4xi1> -> vector<4x8xf32>
// CHECK:   %[[CAST:.*]] = vector.shape_cast %[[LOAD]] : vector<4x8xf32> to vector<32xf32>
// CHECK:   return %[[CAST]]
func.func @load_step_offsets(%ptr: i64) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// max-chunk-size = 4: factor 4 -> 8 lanes, chunk_size = 4.
// CHECK4-LABEL: @load_step_offsets_chunk4
// CHECK4: xegpu.load {{.*}} <{chunk_size = 4 : i64}>
// CHECK4-SAME: : i64, vector<8xindex>, vector<8xi1> -> vector<8x4xf32>
// CHECK4: vector.shape_cast {{.*}} : vector<8x4xf32> to vector<32xf32>
func.func @load_step_offsets_chunk4(%ptr: i64) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// Dense constant arithmetic progression with stride 1.
// CHECK-LABEL: @load_dense_ap_offsets
// CHECK: %[[CST:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
// CHECK: xegpu.load {{.*}}[%[[CST]]], {{.*}} <{chunk_size = 8 : i64}>
// CHECK-SAME: : i64, vector<4xindex>, vector<4xi1> -> vector<4x8xi32>
// CHECK: vector.shape_cast {{.*}} : vector<4x8xi32> to vector<32xi32>
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
// CHECK-LABEL: @load_broadcast_offsets
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
// Non-stride-1 (stride 4) offsets: not contiguous, not coalesced.
// CHECK-LABEL: @load_stride4_unchanged
// CHECK: xegpu.load
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
// CHECK-LABEL: @load_partial_mask_unchanged
// CHECK: xegpu.load
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
func.func @load_partial_mask_unchanged(%ptr: i64, %mask: vector<32xi1>) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %v = xegpu.load %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// Store with vector.step offsets coalesces.
// CHECK-LABEL: @store_step_offsets
// CHECK: vector.shape_cast {{.*}} : vector<32xf32> to vector<4x8xf32>
// CHECK: xegpu.store
// CHECK-SAME: <{chunk_size = 8 : i64}>
// CHECK-SAME: vector<4x8xf32>, i64, vector<4xindex>, vector<4xi1>
func.func @store_step_offsets(%ptr: i64, %v: vector<32xf32>) {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  xegpu.store %v, %ptr[%offsets], %mask <{chunk_size = 1 : i64}>
      : vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
  return
}

// -----
// Store with all-equal offsets is left alone (ambiguous semantics).
// CHECK-LABEL: @store_broadcast_offsets_unchanged
// CHECK: xegpu.store
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
// CHECK-LABEL: @load_memref_step
// CHECK: xegpu.load
// CHECK-SAME: <{chunk_size = 8 : i64}>
// CHECK-SAME: : memref<1024xf32>, vector<4xindex>, vector<4xi1> -> vector<4x8xf32>
// CHECK: vector.shape_cast {{.*}} : vector<4x8xf32> to vector<32xf32>
func.func @load_memref_step(%m: memref<1024xf32>) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %m[%offsets], %mask <{chunk_size = 1 : i64}>
      : memref<1024xf32>, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}
