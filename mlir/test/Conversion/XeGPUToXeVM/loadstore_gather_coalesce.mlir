// RUN: mlir-opt %s --split-input-file -convert-xegpu-to-xevm -canonicalize | FileCheck %s

// Lane-level gather/scatter carry one offset and one mask bit per element. When
// the offsets are one contiguous run and the mask is uniform, the access is
// coalesced into a single block load/store from the base of the run.

gpu.module @test {
// CHECK-LABEL: @load_gather_contiguous_run
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: index, %[[ARG2:.*]]: i1
gpu.func @load_gather_contiguous_run(%src: i64, %base: index, %pred: i1) -> vector<4xf32> {
  %mask = vector.broadcast %pred : i1 to vector<4xi1>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %o1 = arith.addi %base, %c1 : index
  %o2 = arith.addi %base, %c2 : index
  %o3 = arith.addi %base, %c3 : index
  %offsets = vector.from_elements %base, %o1, %o2, %o3 : vector<4xindex>
  // The run base is element 0 of the offsets, and the whole block is gated on
  // one mask bit.
  // CHECK: %[[C4:.*]] = arith.constant 4 : i64
  // CHECK: %[[OFFSETS:.*]] = vector.from_elements %[[ARG1]],
  // CHECK: %[[I64:.*]] = vector.bitcast %[[OFFSETS]] : vector<4xindex> to vector<4xi64>
  // CHECK: %[[BASE:.*]] = vector.extract %[[I64]][0] : i64 from vector<4xi64>
  // CHECK: %[[BYTES:.*]] = arith.muli %[[BASE]], %[[C4]] : i64
  // CHECK: %[[ADDR:.*]] = arith.addi %[[ARG0]], %[[BYTES]] : i64
  // CHECK: %[[PTR:.*]] = llvm.inttoptr %[[ADDR]] : i64 to !llvm.ptr<1>
  // CHECK: scf.if %[[ARG2]] -> (vector<4xf32>) {
  // CHECK:   llvm.load %[[PTR]] {{.*}} : !llvm.ptr<1> -> vector<4xf32>
  %0 = xegpu.load %src[%offsets], %mask <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : i64, vector<4xindex>, vector<4xi1> -> vector<4xf32>
  gpu.return %0 : vector<4xf32>
}
}

// -----

gpu.module @test {
// CHECK-LABEL: @store_scatter_contiguous_run
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: index, %[[ARG2:.*]]: vector<4xf32>
gpu.func @store_scatter_contiguous_run(%dst: i64, %base: index, %val: vector<4xf32>) {
  %mask = arith.constant dense<true> : vector<4xi1>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %o1 = arith.addi %base, %c1 : index
  %o2 = arith.addi %base, %c2 : index
  %o3 = arith.addi %base, %c3 : index
  %offsets = vector.from_elements %base, %o1, %o2, %o3 : vector<4xindex>
  // An all-true mask leaves no scf.if at all.
  // CHECK-NOT: scf.if
  // CHECK: %[[BASE:.*]] = vector.extract %{{.*}}[0] : i64 from vector<4xi64>
  // CHECK: %[[PTR:.*]] = llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<1>
  // CHECK: llvm.store %[[ARG2]], %[[PTR]] {{.*}} : vector<4xf32>, !llvm.ptr<1>
  xegpu.store %val, %dst[%offsets], %mask <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
      : vector<4xf32>, i64, vector<4xindex>, vector<4xi1>
  gpu.return
}
}

// -----

// vector.step is a contiguous run based at 0.

gpu.module @test {
// CHECK-LABEL: @load_gather_step_offsets
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i1
gpu.func @load_gather_step_offsets(%src: i64, %pred: i1) -> vector<4xf32> {
  %offsets = vector.step : vector<4xindex>
  %mask = vector.broadcast %pred : i1 to vector<4xi1>
  // CHECK: %[[STEP:.*]] = vector.step : vector<4xindex>
  // CHECK: %[[I64:.*]] = vector.bitcast %[[STEP]] : vector<4xindex> to vector<4xi64>
  // CHECK: %[[BASE:.*]] = vector.extract %[[I64]][0] : i64 from vector<4xi64>
  // CHECK: %[[PTR:.*]] = llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<1>
  // CHECK: scf.if %[[ARG1]] -> (vector<4xf32>) {
  // CHECK:   llvm.load %[[PTR]] {{.*}} : !llvm.ptr<1> -> vector<4xf32>
  %0 = xegpu.load %src[%offsets], %mask <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : i64, vector<4xindex>, vector<4xi1> -> vector<4xf32>
  gpu.return %0 : vector<4xf32>
}
}

// -----

// A mask built from one repeated value is uniform too.

gpu.module @test {
// CHECK-LABEL: @load_gather_from_elements_mask
gpu.func @load_gather_from_elements_mask(%src: i64, %base: index, %pred: i1) -> vector<2xf32> {
  %c1 = arith.constant 1 : index
  %o1 = arith.addi %base, %c1 : index
  %offsets = vector.from_elements %base, %o1 : vector<2xindex>
  %mask = vector.from_elements %pred, %pred : vector<2xi1>
  // CHECK: %[[BASE:.*]] = vector.extract %{{.*}}[0] : i64 from vector<2xi64>
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<1> -> vector<2xf32>
  %0 = xegpu.load %src[%offsets], %mask <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : i64, vector<2xindex>, vector<2xi1> -> vector<2xf32>
  gpu.return %0 : vector<2xf32>
}
}
