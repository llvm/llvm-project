// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=1 memref-load-bitwidth=32 skip-memref-type-conversion" --split-input-file %s | FileCheck %s

// These tests mimic tests from vector-narrow-type.mlir, but load/store 2-D
// insted of 1-D vectors. That's currently not supported.

///----------------------------------------------------------------------------------------
/// vector.load
///----------------------------------------------------------------------------------------

func.func @vector_load_2d_i8_negative(%arg1: index, %arg2: index) -> vector<2x4xi8> {
    %0 = memref.alloc() : memref<3x4xi8>
    %1 = vector.load %0[%arg1, %arg2] : memref<3x4xi8>, vector<2x4xi8>
    return %1 : vector<2x4xi8>
}

// No support for loading 2D vectors - expect no conversions
//  CHECK-LABEL: func @vector_load_2d_i8_negative
//        CHECK:   memref.alloc() : memref<3x4xi8>
//    CHECK-NOT: i32

// -----

///----------------------------------------------------------------------------------------
/// vector.transfer_read
///----------------------------------------------------------------------------------------

func.func @vector_transfer_read_2d_i4_negative(%arg1: index, %arg2: index) -> vector<2x8xi4> {
    %c0 = arith.constant 0 : i4
    %0 = memref.alloc() : memref<3x8xi4>
    %1 = vector.transfer_read %0[%arg1, %arg2], %c0 {in_bounds = [true, true]} :
      memref<3x8xi4>, vector<2x8xi4>
    return %1 : vector<2x8xi4>
}
//  CHECK-LABEL: func @vector_transfer_read_2d_i4_negative
//        CHECK:  memref.alloc() : memref<3x8xi4>
//    CHECK-NOT: i32

// -----

///----------------------------------------------------------------------------------------
/// vector.maskedload
///----------------------------------------------------------------------------------------

func.func @vector_maskedload_2d_i8_negative(%arg1: index, %arg2: index, %arg3: index, %passthru: vector<2x4xi8>) -> vector<2x4xi8> {
    %0 = memref.alloc() : memref<3x4xi8>
    %mask = vector.create_mask %arg3, %arg3 : vector<2x4xi1>
    %1 = vector.maskedload %0[%arg1, %arg2], %mask, %passthru :
      memref<3x4xi8>, vector<2x4xi1>, vector<2x4xi8> into vector<2x4xi8>
    return %1 : vector<2x4xi8>
}

//  CHECK-LABEL: func @vector_maskedload_2d_i8_negative
//        CHECK:  memref.alloc() : memref<3x4xi8>
//    CHECK-NOT: i32

// -----

///----------------------------------------------------------------------------------------
/// vector.extract -> vector.masked_load
///----------------------------------------------------------------------------------------

func.func @vector_extract_maskedload_2d_i4_negative(%arg1: index) -> vector<8x8x16xi4> {
    %0 = memref.alloc() : memref<8x8x16xi4>
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %cst_1 = arith.constant dense<0> : vector<8x8x16xi4>
    %cst_2 = arith.constant dense<0> : vector<8x16xi4>
    %27 = vector.create_mask %c8, %arg1, %c16 : vector<8x8x16xi1>
    %48 = vector.extract %27[0] : vector<8x16xi1> from vector<8x8x16xi1>
    %50 = vector.maskedload %0[%c0, %c0, %c0], %48, %cst_2 : memref<8x8x16xi4>, vector<8x16xi1>, vector<8x16xi4> into vector<8x16xi4>
    %63 = vector.insert %50, %cst_1 [0] : vector<8x16xi4> into vector<8x8x16xi4>
    return %63 : vector<8x8x16xi4>
}

//  CHECK-LABEL: func @vector_extract_maskedload_2d_i4_negative
//        CHECK: memref.alloc() : memref<8x8x16xi4>
//    CHECK-NOT: i32

// -----

///----------------------------------------------------------------------------------------
/// vector.store
///----------------------------------------------------------------------------------------

func.func @vector_store_2d_i8_negative(%arg0: vector<2x8xi8>, %arg1: index, %arg2: index) {
    %0 = memref.alloc() : memref<4x8xi8>
    vector.store %arg0, %0[%arg1, %arg2] :memref<4x8xi8>, vector<2x8xi8>
    return
}

//  CHECK-LABEL: func @vector_store_2d_i8_negative
//        CHECK: memref.alloc() : memref<4x8xi8>
//    CHECK-NOT: i32

// -----

///----------------------------------------------------------------------------------------
/// vector.maskedstore
///----------------------------------------------------------------------------------------

func.func @vector_maskedstore_2d_i8_negative(%arg0: index, %arg1: index, %arg2: index, %value: vector<2x8xi8>) {
  %0 = memref.alloc() : memref<3x8xi8>
  %mask = vector.create_mask %arg2, %arg2 : vector<2x8xi1>
  vector.maskedstore %0[%arg0, %arg1], %mask, %value : memref<3x8xi8>, vector<2x8xi1>, vector<2x8xi8>
  return
}

//  CHECK-LABEL: func @vector_maskedstore_2d_i8_negative
//        CHECK: memref.alloc() : memref<3x8xi8>
//    CHECK-NOT: i32
