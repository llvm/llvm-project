// RUN: mlir-opt %s -test-vector-to-vector-lowering | FileCheck %s

//-----------------------------------------------------------------------------
// [Pattern: MaskedLoadFolder]
//-----------------------------------------------------------------------------

// CHECK-LABEL:   func @fold_maskedload_all_true_dynamic(
// CHECK-SAME:                      %[[BASE:.*]]: memref<?xf32>,
// CHECK-SAME:                      %[[PASS_THRU:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-DAG:       %[[IDX:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[LOAD:.*]] = vector.load %[[BASE]][%[[IDX]]] : memref<?xf32>, vector<16xf32>
// CHECK-NEXT:      return %[[LOAD]] : vector<16xf32>
func.func @fold_maskedload_all_true_dynamic(%base: memref<?xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.maskedload %base[%c0], %mask, %pass_thru
    : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL:   func @fold_maskedload_all_true_static(
// CHECK-SAME:                      %[[BASE:.*]]: memref<16xf32>,
// CHECK-SAME:                      %[[PASS_THRU:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-DAG:       %[[IDX:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[LOAD:.*]] = vector.load %[[BASE]][%[[IDX]]] : memref<16xf32>, vector<16xf32>
// CHECK-NEXT:      return %[[LOAD]] : vector<16xf32>
func.func @fold_maskedload_all_true_static(%base: memref<16xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.maskedload %base[%c0], %mask, %pass_thru
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL:   func @fold_maskedload_all_false_static(
// CHECK-SAME:                      %[[BASE:.*]]: memref<16xf32>,
// CHECK-SAME:                      %[[PASS_THRU:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-NEXT:      return %[[PASS_THRU]] : vector<16xf32>
func.func @fold_maskedload_all_false_static(%base: memref<16xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [0] : vector<16xi1>
  %ld = vector.maskedload %base[%c0], %mask, %pass_thru
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL:   func @fold_maskedload_dynamic_non_zero_idx(
// CHECK-SAME:                      %[[BASE:.*]]: memref<?xf32>,
// CHECK-SAME:                      %[[PASS_THRU:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-DAG:       %[[IDX:.*]] = arith.constant 8 : index
// CHECK-NEXT:      %[[LOAD:.*]] = vector.load %[[BASE]][%[[IDX]]] : memref<?xf32>, vector<16xf32>
// CHECK-NEXT:      return %[[LOAD]] : vector<16xf32>
func.func @fold_maskedload_dynamic_non_zero_idx(%base: memref<?xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c8 = arith.constant 8 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.maskedload %base[%c8], %mask, %pass_thru
    : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

//-----------------------------------------------------------------------------
// [Pattern: MaskedStoreFolder]
//-----------------------------------------------------------------------------

// CHECK-LABEL:   func @fold_maskedstore_all_true(
// CHECK-SAME:                       %[[BASE:.*]]: memref<16xf32>,
// CHECK-SAME:                       %[[VALUE:.*]]: vector<16xf32>) {
// CHECK-NEXT:      %[[IDX:.*]] = arith.constant 0 : index
// CHECK-NEXT:      vector.store %[[VALUE]], %[[BASE]][%[[IDX]]] : memref<16xf32>, vector<16xf32>
// CHECK-NEXT:      return
func.func @fold_maskedstore_all_true(%base: memref<16xf32>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  vector.maskedstore %base[%c0], %mask, %value : memref<16xf32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL:   func @fold_maskedstore_all_false(
// CHECK-SAME:                       %[[BASE:.*]]: memref<16xf32>,
// CHECK-SAME:                       %[[VALUE:.*]]: vector<16xf32>) {
// CHECK-NEXT:      return
func.func @fold_maskedstore_all_false(%base: memref<16xf32>, %value: vector<16xf32>)  {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [0] : vector<16xi1>
  vector.maskedstore %base[%c0], %mask, %value : memref<16xf32>, vector<16xi1>, vector<16xf32>
  return
}

//-----------------------------------------------------------------------------
// [Pattern: GatherFolder]
//-----------------------------------------------------------------------------

/// There is no alternative (i.e. simpler) Op for this, hence no-fold.

// CHECK-LABEL:   func @no_fold_gather_all_true(
// CHECK-SAME:                  %[[BASE:.*]]: memref<16xf32>,
// CHECK-SAME:                  %[[INDICES:.*]]: vector<16xi32>,
// CHECK-SAME:                  %[[PASS_THRU:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-NEXT:      %[[C:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[M:.*]] = arith.constant dense<true> : vector<16xi1>
// CHECK-NEXT:      %[[G:.*]] = vector.gather %[[BASE]][%[[C]]] [%[[INDICES]]], %[[M]], %[[PASS_THRU]] : memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-NEXT:      return %[[G]] : vector<16xf32>
func.func @no_fold_gather_all_true(%base: memref<16xf32>, %indices: vector<16xi32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.gather %base[%c0][%indices], %mask, %pass_thru
    : memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL:   func @fold_gather_all_true(
// CHECK-SAME:                  %[[BASE:.*]]: memref<16xf32>,
// CHECK-SAME:                  %[[INDICES:.*]]: vector<16xi32>,
// CHECK-SAME:                  %[[PASS_THRU:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-NEXT:      return %[[PASS_THRU]] : vector<16xf32>
func.func @fold_gather_all_true(%base: memref<16xf32>, %indices: vector<16xi32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [0] : vector<16xi1>
  %ld = vector.gather %base[%c0][%indices], %mask, %pass_thru
    : memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

//-----------------------------------------------------------------------------
// [Pattern: ScatterFolder]
//-----------------------------------------------------------------------------

/// There is no alternative (i.e. simpler) Op for this, hence no-fold.

// CHECK-LABEL:   func @no_fold_scatter_all_true(
// CHECK-SAME:                   %[[BASE:.*]]: memref<16xf32>,
// CHECK-SAME:                   %[[INDICES:.*]]: vector<16xi32>,
// CHECK-SAME:                   %[[VALUE:.*]]: vector<16xf32>) {
// CHECK-NEXT:      %[[C:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[M:.*]] = arith.constant dense<true> : vector<16xi1>
// CHECK-NEXT:      vector.scatter %[[BASE]][%[[C]]] [%[[INDICES]]], %[[M]], %[[VALUE]] : memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
// CHECK-NEXT:      return
func.func @no_fold_scatter_all_true(%base: memref<16xf32>, %indices: vector<16xi32>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  vector.scatter %base[%c0][%indices], %mask, %value
    : memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL:   func @fold_scatter_all_false(
// CHECK-SAME:                   %[[BASE:.*]]: memref<16xf32>,
// CHECK-SAME:                   %[[INDICES:.*]]: vector<16xi32>,
// CHECK-SAME:                   %[[VALUE:.*]]: vector<16xf32>) {
// CHECK-NEXT:      return
func.func @fold_scatter_all_false(%base: memref<16xf32>, %indices: vector<16xi32>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %0 = vector.type_cast %base : memref<16xf32> to memref<vector<16xf32>>
  %mask = vector.constant_mask [0] : vector<16xi1>
  vector.scatter %base[%c0][%indices], %mask, %value
    : memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
  return
}

//-----------------------------------------------------------------------------
// [Pattern: ExpandLoadFolder]
//-----------------------------------------------------------------------------

// CHECK-LABEL:   func @fold_expandload_all_true(
// CHECK-SAME:                  %[[BASE:.*]]: memref<16xf32>,
// CHECK-SAME:                  %[[PASS_THRU:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-DAG:       %[[C:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[T:.*]] = vector.load %[[BASE]][%[[C]]] : memref<16xf32>, vector<16xf32>
// CHECK-NEXT:      return %[[T]] : vector<16xf32>
func.func @fold_expandload_all_true(%base: memref<16xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.expandload %base[%c0], %mask, %pass_thru
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL:   func @fold_expandload_all_false(
// CHECK-SAME:                  %[[BASE:.*]]: memref<16xf32>,
// CHECK-SAME:                  %[[PASS_THRU:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-NEXT:      return %[[PASS_THRU]] : vector<16xf32>
func.func @fold_expandload_all_false(%base: memref<16xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [0] : vector<16xi1>
  %ld = vector.expandload %base[%c0], %mask, %pass_thru
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

//-----------------------------------------------------------------------------
// [Pattern: CompressStoreFolder]
//-----------------------------------------------------------------------------

// CHECK-LABEL:   func @fold_compressstore_all_true(
// CHECK-SAME:                    %[[BASE:.*]]: memref<16xf32>,
// CHECK-SAME:                    %[[VALUE:.*]]: vector<16xf32>) {
// CHECK-NEXT:      %[[C:.*]] = arith.constant 0 : index
// CHECK-NEXT:      vector.store %[[VALUE]], %[[BASE]][%[[C]]] : memref<16xf32>, vector<16xf32>
// CHECK-NEXT:      return
func.func @fold_compressstore_all_true(%base: memref<16xf32>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  vector.compressstore %base[%c0], %mask, %value  : memref<16xf32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL:   func @fold_compressstore_all_false(
// CHECK-SAME:                    %[[BASE:.*]]: memref<16xf32>,
// CHECK-SAME:                    %[[VALUE:.*]]: vector<16xf32>) {
// CHECK-NEXT:      return
func.func @fold_compressstore_all_false(%base: memref<16xf32>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [0] : vector<16xi1>
  vector.compressstore %base[%c0], %mask, %value : memref<16xf32>, vector<16xi1>, vector<16xf32>
  return
}
