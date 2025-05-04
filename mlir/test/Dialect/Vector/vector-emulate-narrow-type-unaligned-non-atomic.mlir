// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=1 memref-load-bitwidth=8 disable-atomic-rmw=true" --cse --split-input-file %s | FileCheck %s

// NOTE: In this file all RMW stores are non-atomic.

// TODO: remove memref.alloc() in the tests to eliminate noises.
// memref.alloc exists here because sub-byte vector data types such as i2
// are currently not supported as input arguments.

///----------------------------------------------------------------------------------------
/// vector.store
///----------------------------------------------------------------------------------------

func.func @vector_store_i2_const_index_two_partial_stores(%src: vector<3xi2>) {
    %dest = memref.alloc() : memref<3x3xi2>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    vector.store %src, %dest[%c2, %c0] :memref<3x3xi2>, vector<3xi2>
    return
}

//  Store 6 bits from the input vector into bytes [1:2] of a 3-byte destination
//  memref, i.e. into  bits [12:18) of a 24-bit destintion container
//  (`memref<3x3xi2>` is emulated via `memref<3xi8>`). This requires two
//  non-atomic RMW partial stores. Due to partial storing, both bytes are
//  accessed partially through masking.

//      CHECK:  func @vector_store_i2_const_index_two_partial_stores(
// CHECK-SAME:    %[[SRC:.+]]: vector<3xi2>)

//      CHECK:  %[[DEST:.+]] = memref.alloc() : memref<3xi8>
//      CHECK:  %[[C1:.+]] = arith.constant 1 : index

// RMW sequence for Byte 1
//      CHECK:  %[[MASK_1:.+]] = arith.constant dense<[false, false, true, true]>
//      CHECK:  %[[INIT:.+]] = arith.constant dense<0> : vector<4xi2>
//      CHECK:  %[[SRC_SLICE_1:.+]] = vector.extract_strided_slice %[[SRC]]
// CHECK-SAME:    {offsets = [0], sizes = [2], strides = [1]} : vector<3xi2> to vector<2xi2>
//      CHECK:  %[[INIT_WITH_SLICE_1:.+]] = vector.insert_strided_slice %[[SRC_SLICE_1]], %[[INIT]]
// CHECK-SAME:    {offsets = [2], strides = [1]} : vector<2xi2> into vector<4xi2>
//      CHECK:  %[[DEST_BYTE_1:.+]] = vector.load %[[DEST]][%[[C1]]] : memref<3xi8>, vector<1xi8>
//      CHECK:  %[[DEST_BYTE_1_AS_I2:.+]] = vector.bitcast %[[DEST_BYTE_1]]
// CHECK-SAME:    vector<1xi8> to vector<4xi2>
//      CHECK:  %[[RES_BYTE_1:.+]] = arith.select %[[MASK_1]], %[[INIT_WITH_SLICE_1]], %[[DEST_BYTE_1_AS_I2]]
//      CHECK:  %[[RES_BYTE_1_AS_I8:.+]] = vector.bitcast %[[RES_BYTE_1]]
// CHECK-SAME:    vector<4xi2> to vector<1xi8>
//      CHECK:  vector.store %[[RES_BYTE_1_AS_I8]], %[[DEST]][%[[C1]]]

// RMW sequence for Byte 2
//      CHECK:  %[[OFFSET:.+]] = arith.addi %[[C1]], %[[C1]] : index
//      CHECK:  %[[SRC_SLICE_2:.+]] = vector.extract_strided_slice %[[SRC]]
// CHECK-SAME:    {offsets = [2], sizes = [1], strides = [1]} : vector<3xi2> to vector<1xi2>
//      CHECK:  %[[INIT_WITH_SLICE_2:.+]] = vector.insert_strided_slice %[[SRC_SLICE_2]], %[[INIT]]
// CHECK-SAME:    {offsets = [0], strides = [1]} : vector<1xi2> into vector<4xi2>
//      CHECK:  %[[MASK_2:.+]] = arith.constant dense<[true, false, false, false]> : vector<4xi1>
//      CHECK:  %[[DEST_BYTE_2:.+]] = vector.load %[[DEST]][%[[OFFSET]]] : memref<3xi8>, vector<1xi8>
//      CHECK:  %[[DEST_BYTE_2_AS_I2:.+]] = vector.bitcast %[[DEST_BYTE_2]]
// CHECK-SAME:    vector<1xi8> to vector<4xi2>
//      CHECK:  %[[RES_BYTE_2:.+]] = arith.select %[[MASK_2]], %[[INIT_WITH_SLICE_2]], %[[DEST_BYTE_2_AS_I2]]
//      CHECK:  %[[RES_BYTE_2_AS_I8:.+]] = vector.bitcast %[[RES_BYTE_2]]
// CHECK-SAME:    vector<4xi2> to vector<1xi8>
//      CHECK:  vector.store %[[RES_BYTE_2_AS_I8]], %[[DEST]][%[[OFFSET]]]

// -----

func.func @vector_store_i2_two_partial_one_full_stores(%src: vector<7xi2>) {
    %dest = memref.alloc() : memref<3x7xi2>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    vector.store %src, %dest[%c1, %c0] :memref<3x7xi2>, vector<7xi2>
    return
}

// Store 14 bits from the input vector into bytes [1:3] of a 6-byte destination
// memref, i.e. bits [15:29) of a 48-bit destination container memref
// (`memref<3x7xi2>` is emulated via `memref<6xi8>`). This requires two
// non-atomic RMW stores (for the "boundary" bytes) and one full byte store
// (for the "middle" byte). Note that partial stores require masking.

//      CHECK: func @vector_store_i2_two_partial_one_full_stores(
// CHECK-SAME:    %[[SRC:.+]]:

//      CHECK:  %[[DEST:.+]] = memref.alloc() : memref<6xi8>
//      CHECK:  %[[C1:.+]] = arith.constant 1 : index

// First partial/RMW store:
//      CHECK:  %[[MASK_1:.+]] = arith.constant dense<[false, false, false, true]>
//      CHECK:  %[[INIT:.+]] = arith.constant dense<0> : vector<4xi2>
//      CHECK:  %[[SRC_SLICE_0:.+]] = vector.extract_strided_slice %[[SRC]]
// CHECK-SAME:    {offsets = [0], sizes = [1], strides = [1]}
//      CHECK:  %[[INIT_WITH_SLICE_1:.+]] = vector.insert_strided_slice %[[SRC_SLICE_0]], %[[INIT]]
// CHECK-SAME:    {offsets = [3], strides = [1]}
//      CHECK:  %[[DEST_BYTE_1:.+]] = vector.load %[[DEST]][%[[C1]]]
//      CHECK:  %[[DEST_BYTE_AS_I2:.+]] = vector.bitcast %[[DEST_BYTE_1]]
// CHECK-SAME:    : vector<1xi8> to vector<4xi2>
//      CHECK:  %[[RES_BYTE_1:.+]] = arith.select %[[MASK_1]], %[[INIT_WITH_SLICE_1]], %[[DEST_BYTE_AS_I2]]
//      CHECK:  %[[RES_BYTE_1_AS_I8:.+]] = vector.bitcast %[[RES_BYTE_1]]
// CHECK-SAME:    : vector<4xi2> to vector<1xi8>
//      CHECK:  vector.store %[[RES_BYTE_1_AS_I8]], %[[DEST]][%[[C1]]]

// Full-width store:
//      CHECK:  %[[C2:.+]] = arith.addi %[[C1]], %[[C1]]
//      CHECK:  %[[SRC_SLICE_1:.+]] = vector.extract_strided_slice %[[SRC]]
// CHECK-SAME:    {offsets = [1], sizes = [4], strides = [1]}
//      CHECK:  %[[SRC_SLICE_1_AS_I8:.+]] = vector.bitcast %[[SRC_SLICE_1]]
// CHECK-SAME:    : vector<4xi2> to vector<1xi8>
//      CHECK:  vector.store %[[SRC_SLICE_1_AS_I8]], %[[DEST]][%[[C2]]]

// Second partial/RMW store:
//      CHECK:  %[[C3:.+]] = arith.addi %[[C2]], %[[C1]]
//      CHECK:  %[[SRC_SLICE_2:.+]] = vector.extract_strided_slice %[[SRC]]
// CHECK-SAME:    {offsets = [5], sizes = [2], strides = [1]}
//      CHECK:  %[[INIT_WITH_SLICE2:.+]] = vector.insert_strided_slice %[[SRC_SLICE_2]]
// CHECK-SAME:    {offsets = [0], strides = [1]}
//      CHECK:  %[[MASK_2:.+]] = arith.constant dense<[true, true, false, false]>
//      CHECK:  %[[DEST_BYTE_2:.+]] = vector.load %[[DEST]][%[[C3]]]
//      CHECK:  %[[DEST_BYTE_2_AS_I2:.+]] = vector.bitcast %[[DEST_BYTE_2]]
//      CHECK:  %[[RES_BYTE_2:.+]] = arith.select %[[MASK_2]], %[[INIT_WITH_SLICE2]], %[[DEST_BYTE_2_AS_I2]]
//      CHECK:  %[[RES_BYTE_2_AS_I8:.+]] = vector.bitcast %[[RES_BYTE_2]]
// CHECK-SAME:    : vector<4xi2> to vector<1xi8>
//      CHECK:  vector.store %[[RES_BYTE_2_AS_I8]], %[[DEST]][%[[C3]]]

// -----

func.func @vector_store_i2_const_index_one_partial_store(%src: vector<1xi2>) {
    %dest = memref.alloc() : memref<4x1xi2>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    vector.store %src, %dest[%c1, %c0] :memref<4x1xi2>, vector<1xi2>
    return
}

// Store 2 bits from the input vector into byte 0 of a 1-byte destination
// memref, i.e. bits [3:5) of a 8-bit destination container memref
// (`<memref<4x1xi2>` is emulated via `memref<1xi8>`). This requires one
// non-atomic RMW.

//      CHECK:  func @vector_store_i2_const_index_one_partial_store(
// CHECK-SAME:    %[[SRC:.+]]: vector<1xi2>)

//      CHECK:  %[[DEST:.+]] = memref.alloc() : memref<1xi8>
//      CHECK:  %[[C0:.+]] = arith.constant 0 : index

//      CHECK:  %[[MASK:.+]] = arith.constant dense<[false, true, false, false]>
//      CHECK:  %[[INIT:.+]] = arith.constant dense<0> : vector<4xi2>
//      CHECK:  %[[INIT_WITH_SLICE:.+]] = vector.insert_strided_slice %[[SRC]], %[[INIT]]
// CHECK-SAME:    {offsets = [1], strides = [1]} : vector<1xi2> into vector<4xi2>
//      CHECK:  %[[DEST_BYTE:.+]] = vector.load %[[DEST]][%[[C0]]] : memref<1xi8>, vector<1xi8>
//      CHECK:  %[[DEST_BYTE_AS_I2:.+]] = vector.bitcast %[[DEST_BYTE]]
// CHECK-SAME:    : vector<1xi8> to vector<4xi2>
//      CHECK:  %[[RES_BYTE:.+]] = arith.select %[[MASK]], %[[INIT_WITH_SLICE]], %[[DEST_BYTE_AS_I2]]
//      CHECK:  %[[RES_BYTE_AS_I8:.+]] = vector.bitcast %[[RES_BYTE]]
// CHECK-SAME:    : vector<4xi2> to vector<1xi8>
//      CHECK:  vector.store %[[RES_BYTE_AS_I8]], %[[DEST]][%[[C0]]]
