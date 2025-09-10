// RUN: mlir-opt --test-memref-flatten-and-vector-narrow-type-emulation --split-input-file %s | FileCheck %s

// -----

func.func @vector_load_2d_i4(%arg0: index, %arg1: index) -> vector<8xi4> {
    %0 = memref.alloc() : memref<4x8xi4>
    %1 = vector.load %0[%arg0, %arg1] : memref<4x8xi4>, vector<8xi4>
    return %1 : vector<8xi4>
}
//      CHECK: func @vector_load_2d_i4
//      CHECK:   vector.load
// CHECK-SAME:   memref<16xi8>

// -----

func.func @vector_maskedload_2d_i4(%arg0: index, %arg1: index, %passthru: vector<8xi4>) -> vector<8xi4> {
    %0 = memref.alloc() : memref<4x8xi4>
    %mask = vector.constant_mask [6] : vector<8xi1>
    %1 = vector.maskedload %0[%arg0, %arg1], %mask, %passthru :
      memref<4x8xi4>, vector<8xi1>, vector<8xi4> into vector<8xi4>
    return %1 : vector<8xi4>
}
//      CHECK: func @vector_maskedload_2d_i4(
//      CHECK:   vector.maskedload
// CHECK-SAME:   memref<16xi8>

// -----

func.func @vector_maskedstore_2d_i4(%arg0: index, %arg1: index, %value: vector<8xi4>) {
    %0 = memref.alloc() : memref<4x8xi4>
    %mask = vector.constant_mask [5] : vector<8xi1>
    vector.maskedstore %0[%arg0, %arg1], %mask, %value :
      memref<4x8xi4>, vector<8xi1>, vector<8xi4>
    return
}
//      CHECK: func @vector_maskedstore_2d_i4(
//      CHECK:   vector.maskedstore
// CHECK-SAME:   memref<16xi8>
