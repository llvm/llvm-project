// RUN: mlir-opt %s -split-input-file -xegpu-canonicalize | FileCheck %s

// CHECK-LABEL: @gather_2d_from_flat
// CHECK-SAME:    %[[SRC:.+]]: memref<?xbf16, strided<[1], offset: ?>>,
// CHECK-SAME:    %[[IDX:.+]]: vector<128x64xindex>, %[[MASK:.+]]: vector<128x64xi1>
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[PASS_THRU:.+]] = arith.constant dense<0.000000e+00> : vector<128x64xbf16>
// CHECK-NOT:     vector.shape_cast
// CHECK:         %[[RES:.+]] = vector.gather %[[SRC]][%[[C0]]] [%[[IDX]]], %[[MASK]], %[[PASS_THRU]]
// CHECK-SAME:      into vector<128x64xbf16>
// CHECK-NOT:     vector.shape_cast
// CHECK:         return %[[RES]] : vector<128x64xbf16>
func.func @gather_2d_from_flat(%src: memref<?xbf16, strided<[1], offset: ?>>,
    %idx: vector<128x64xindex>, %mask: vector<128x64xi1>) -> vector<128x64xbf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<8192xbf16>
  %flat_idx = vector.shape_cast %idx : vector<128x64xindex> to vector<8192xindex>
  %flat_mask = vector.shape_cast %mask : vector<128x64xi1> to vector<8192xi1>
  %flat_res = vector.gather %src[%c0] [%flat_idx], %flat_mask, %cst
    : memref<?xbf16, strided<[1], offset: ?>>, vector<8192xindex>, vector<8192xi1>,
      vector<8192xbf16> into vector<8192xbf16>
  %res = vector.shape_cast %flat_res : vector<8192xbf16> to vector<128x64xbf16>
  return %res : vector<128x64xbf16>
}

// -----

// CHECK-LABEL: @scatter_2d_from_flat
// CHECK-SAME:    %[[SRC:.+]]: memref<?xbf16, strided<[1], offset: ?>>,
// CHECK-SAME:    %[[IDX:.+]]: vector<128x128xindex>, %[[MASK:.+]]: vector<128x128xi1>,
// CHECK-SAME:    %[[VAL:.+]]: vector<128x128xbf16>
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK-NOT:     vector.shape_cast
// CHECK:         vector.scatter %[[SRC]][%[[C0]]] [%[[IDX]]], %[[MASK]], %[[VAL]]
func.func @scatter_2d_from_flat(%src: memref<?xbf16, strided<[1], offset: ?>>,
    %idx: vector<128x128xindex>, %mask: vector<128x128xi1>,
    %val: vector<128x128xbf16>) {
  %c0 = arith.constant 0 : index
  %flat_idx = vector.shape_cast %idx : vector<128x128xindex> to vector<16384xindex>
  %flat_mask = vector.shape_cast %mask : vector<128x128xi1> to vector<16384xi1>
  %flat_val = vector.shape_cast %val : vector<128x128xbf16> to vector<16384xbf16>
  vector.scatter %src[%c0] [%flat_idx], %flat_mask, %flat_val
    : memref<?xbf16, strided<[1], offset: ?>>, vector<16384xindex>, vector<16384xi1>,
      vector<16384xbf16>
  return
}

// -----

// A splat mask / pass-thru is rebuilt at the N-D shape.
// CHECK-LABEL: @gather_2d_splat_operands
// CHECK-SAME:    %[[SRC:.+]]: memref<?xf32>, %[[IDX:.+]]: vector<8x16xindex>, %[[P:.+]]: i1
// CHECK-DAG:     %[[MASK:.+]] = vector.broadcast %[[P]] : i1 to vector<8x16xi1>
// CHECK-DAG:     %[[PASS_THRU:.+]] = arith.constant dense<1.000000e+00> : vector<8x16xf32>
// CHECK:         vector.gather {{.*}}, %[[MASK]], %[[PASS_THRU]] {{.*}} into vector<8x16xf32>
func.func @gather_2d_splat_operands(%src: memref<?xf32>, %idx: vector<8x16xindex>,
    %p: i1) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.000000e+00> : vector<128xf32>
  %flat_idx = vector.shape_cast %idx : vector<8x16xindex> to vector<128xindex>
  %flat_mask = vector.broadcast %p : i1 to vector<128xi1>
  %flat_res = vector.gather %src[%c0] [%flat_idx], %flat_mask, %cst
    : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
      into vector<128xf32>
  %res = vector.shape_cast %flat_res : vector<128xf32> to vector<8x16xf32>
  return %res : vector<8x16xf32>
}

// -----

// The mask cannot be un-flattened, so the gather is left alone: rewriting it
// would only move the shape_cast from the index to the mask operand.
// CHECK-LABEL: @gather_opaque_mask_untouched
// CHECK:         vector.gather {{.*}} into vector<128xf32>
func.func @gather_opaque_mask_untouched(%src: memref<?xf32>, %idx: vector<8x16xindex>,
    %mask: vector<128xi1>, %pass_thru: vector<128xf32>) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %flat_idx = vector.shape_cast %idx : vector<8x16xindex> to vector<128xindex>
  %flat_res = vector.gather %src[%c0] [%flat_idx], %mask, %pass_thru
    : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
      into vector<128xf32>
  %res = vector.shape_cast %flat_res : vector<128xf32> to vector<8x16xf32>
  return %res : vector<8x16xf32>
}

// -----

// The flat result is consumed as-is, so the gather is left alone: rewriting it
// would only move the shape_casts to the result.
// CHECK-LABEL: @gather_flat_use_untouched
// CHECK:         vector.gather {{.*}} into vector<128xf32>
func.func @gather_flat_use_untouched(%src: memref<?xf32>, %idx: vector<8x16xindex>,
    %mask: vector<8x16xi1>, %pass_thru: vector<128xf32>) -> vector<128xf32> {
  %c0 = arith.constant 0 : index
  %flat_idx = vector.shape_cast %idx : vector<8x16xindex> to vector<128xindex>
  %flat_mask = vector.shape_cast %mask : vector<8x16xi1> to vector<128xi1>
  %res = vector.gather %src[%c0] [%flat_idx], %flat_mask, %pass_thru
    : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
      into vector<128xf32>
  return %res : vector<128xf32>
}
