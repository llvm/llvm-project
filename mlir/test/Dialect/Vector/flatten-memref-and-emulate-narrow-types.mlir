// RUN: mlir-opt --test-memref-flatten-and-vector-narrow-type-emulation --split-input-file %s | FileCheck %s

// This test verifies that narrow-type-emulation works correctly for
// rank > 1 memrefs by combining memref flattening with vector narrow type
// emulation patterns. 
//
// The patterns tested here demonstrate the composition of two transformations,
// memref flattening for vector ops and vector op narrow type emulation.
//
// TODO: Support `vector.transfer_write` operation.

func.func @vector_load_2d_i4(%arg0: index) -> vector<8xi4> {
    %0 = memref.alloc() : memref<4x8xi4>
    %1 = vector.load %0[%arg0, %arg0] : memref<4x8xi4>, vector<8xi4>
    return %1 : vector<8xi4>
}
//  CHECK-LABEL: func @vector_load_2d_i4
//        CHECK:   vector.load {{.*}} memref<16xi8>

// -----

func.func @vector_maskedload_2d_i4(%arg0: index, %passthru: vector<8xi4>) -> vector<8xi4> {
    %0 = memref.alloc() : memref<4x8xi4>
    %mask = vector.constant_mask [6] : vector<8xi1>
    %1 = vector.maskedload %0[%arg0, %arg0], %mask, %passthru :
      memref<4x8xi4>, vector<8xi1>, vector<8xi4> into vector<8xi4>
    return %1 : vector<8xi4>
}
//  CHECK-LABEL: func @vector_maskedload_2d_i4(
//        CHECK:   vector.maskedload {{.*}} memref<16xi8>

// -----

func.func @vector_maskedstore_2d_i4(%arg0: index, %value: vector<8xi4>) {
    %0 = memref.alloc() : memref<4x8xi4>
    %mask = vector.constant_mask [5] : vector<8xi1>
    vector.maskedstore %0[%arg0, %arg0], %mask, %value :
      memref<4x8xi4>, vector<8xi1>, vector<8xi4>
    return
}
//  CHECK-LABEL: func @vector_maskedstore_2d_i4(
//        CHECK:   vector.maskedstore {{.*}} memref<16xi8>

// -----

func.func @vector_store_2d_i4(%arg0: index, %value: vector<8xi4>) {
    %0 = memref.alloc() : memref<4x8xi4>
    vector.store %value, %0[%arg0, %arg0] : memref<4x8xi4>, vector<8xi4>
    return
}
//  CHECK-LABEL: func @vector_store_2d_i4(
//        CHECK:   vector.store {{.*}} memref<16xi8>

// -----

func.func @vector_transfer_read_2d_i4(%arg0: index, %padding: i4) -> vector<8xi4> {
    %0 = memref.alloc() : memref<4x8xi4>
    %1 = vector.transfer_read %0[%arg0, %arg0], %padding {in_bounds = [true]} : memref<4x8xi4>, vector<8xi4>
    return %1 : vector<8xi4>
}
//  CHECK-LABEL: func @vector_transfer_read_2d_i4(
//   CHECK-SAME: %{{.*}}: index, %[[PADDING_I4:.*]]: i4)
//        CHECK:   %[[PADDING_I8:.*]] = arith.extui %[[PADDING_I4]] : i4 to i8
//        CHECK:   vector.transfer_read {{.*}}, %[[PADDING_I8]] : memref<16xi8>, vector<4xi8>
