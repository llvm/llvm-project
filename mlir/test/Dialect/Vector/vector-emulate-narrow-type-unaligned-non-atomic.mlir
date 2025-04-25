// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=1 memref-load-bitwidth=8 disable-atomic-rmw=true" --cse --split-input-file %s | FileCheck %s

// TODO: remove memref.alloc() in the tests to eliminate noises.
// memref.alloc exists here because sub-byte vector data types such as i2
// are currently not supported as input arguments.

///----------------------------------------------------------------------------------------
/// vector.store
///----------------------------------------------------------------------------------------

func.func @vector_store_i2_const_index_two_partial_stores(%arg0: vector<3xi2>) {
    %0 = memref.alloc() : memref<3x3xi2>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    vector.store %arg0, %0[%c2, %c0] :memref<3x3xi2>, vector<3xi2>
    return
}

// Emit two non-atomic RMW partial stores. Store 6 bits from the input vector (bits [12:18)),
// into bytes [1:2] from a 3-byte output memref. Due to partial storing,
// both bytes are accessed partially through masking.

// CHECK: func @vector_store_i2_const_index_two_partial_stores(
// CHECK-SAME: %[[ARG0:.+]]: vector<3xi2>)
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<3xi8>
// CHECK: %[[C1:.+]] = arith.constant 1 : index

// Part 1 RMW sequence
// CHECK: %[[CST:.+]] = arith.constant dense<[false, false, true, true]>
// CHECK: %[[CST0:.+]] = arith.constant dense<0> : vector<4xi2>
// CHECK: %[[EXTRACT:.+]] = vector.extract_strided_slice %[[ARG0]]
// CHECK-SAME: {offsets = [0], sizes = [2], strides = [1]} : vector<3xi2> to vector<2xi2>
// CHECK: %[[INSERT:.+]] = vector.insert_strided_slice %[[EXTRACT]], %[[CST0]]
// CHECK-SAME: {offsets = [2], strides = [1]} : vector<2xi2> into vector<4xi2>
// CHECK: %[[LOAD:.+]] = vector.load
// CHECK: %[[DOWNCAST:.+]] = vector.bitcast %[[LOAD]] : vector<1xi8> to vector<4xi2>
// CHECK: %[[SELECT:.+]] = arith.select %[[CST]], %[[INSERT]], %[[DOWNCAST]]
// CHECK: %[[UPCAST:.+]] = vector.bitcast %[[SELECT]]
// CHECK: vector.store %[[UPCAST]], %[[ALLOC]][%[[C1]]]

// Part 2 RMW sequence
// CHECK: %[[OFFSET:.+]] = arith.addi %[[C1]], %[[C1]] : index
// CHECK: %[[EXTRACT2:.+]] = vector.extract_strided_slice %[[ARG0]]
// CHECK-SAME: {offsets = [2], sizes = [1], strides = [1]} : vector<3xi2> to vector<1xi2>
// CHECK: %[[INSERT2:.+]] = vector.insert_strided_slice %[[EXTRACT2]], %[[CST0]]
// CHECK-SAME: {offsets = [0], strides = [1]} : vector<1xi2> into vector<4xi2>
// CHECK: %[[CST1:.+]] = arith.constant dense<[true, false, false, false]> : vector<4xi1>
// CHECK: %[[LOAD2:.+]] = vector.load
// CHECK: %[[UPCAST2:.+]] = vector.bitcast %[[LOAD2]] : vector<1xi8> to vector<4xi2>
// CHECK: %[[SELECT2:.+]] = arith.select %[[CST1]], %[[INSERT2]], %[[UPCAST2]]
// CHECK: %[[DOWNCAST2:.+]] = vector.bitcast %[[SELECT2]]
// CHECK: vector.store %[[DOWNCAST2]], %[[ALLOC]][%[[OFFSET]]]


// -----

func.func @vector_store_i2_two_partial_one_full_stores(%arg0: vector<7xi2>) {
    %0 = memref.alloc() : memref<3x7xi2>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    vector.store %arg0, %0[%c1, %c0] :memref<3x7xi2>, vector<7xi2>
    return
}

// In this example, emit two RMW stores and one full-width store.

// CHECK: func @vector_store_i2_two_partial_one_full_stores(
// CHECK-SAME: %[[ARG0:.+]]:
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<6xi8>
// CHECK: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[CST:.+]] = arith.constant dense<[false, false, false, true]>
// CHECK: %[[CST0:.+]] = arith.constant dense<0> : vector<4xi2>
// CHECK: %[[EXTRACT:.+]] = vector.extract_strided_slice %[[ARG0]]
// CHECK-SAME: {offsets = [0], sizes = [1], strides = [1]}
// CHECK: %[[INSERT:.+]] = vector.insert_strided_slice %[[EXTRACT]], %[[CST0]]
// CHECK-SAME: {offsets = [3], strides = [1]}
// First sub-width RMW:
// CHECK: %[[LOAD:.+]] = vector.load %[[ALLOC]][%[[C1]]]
// CHECK: %[[UPCAST:.+]] = vector.bitcast %[[LOAD]] : vector<1xi8> to vector<4xi2>
// CHECK: %[[SELECT:.+]] = arith.select %[[CST]], %[[INSERT]], %[[UPCAST]]
// CHECK: %[[DOWNCAST:.+]] = vector.bitcast %[[SELECT]]
// CHECK: vector.store %[[DOWNCAST]], %[[ALLOC]][%[[C1]]]

// Full-width store:
// CHECK: %[[INDEX:.+]] = arith.addi %[[C1]], %[[C1]]
// CHECK: %[[EXTRACT1:.+]] = vector.extract_strided_slice %[[ARG0]]
// CHECK-SAME: {offsets = [1], sizes = [4], strides = [1]}
// CHECK: %[[BITCAST:.+]] = vector.bitcast %[[EXTRACT1]]
// CHECK: vector.store %[[BITCAST]], %[[ALLOC]][%[[INDEX]]]

// Second sub-width RMW:
// CHECK: %[[INDEX2:.+]] = arith.addi %[[INDEX]], %[[C1]]
// CHECK: %[[EXTRACT2:.+]] = vector.extract_strided_slice %[[ARG0]]
// CHECK-SAME: {offsets = [5], sizes = [2], strides = [1]}
// CHECK: %[[INSERT2:.+]] = vector.insert_strided_slice %[[EXTRACT2]]
// CHECK-SAME: {offsets = [0], strides = [1]}
// CHECK: %[[CST1:.+]] = arith.constant dense<[true, true, false, false]>
// CHECK: %[[LOAD2:.+]] = vector.load %[[ALLOC]][%[[INDEX2]]]
// CHECK: %[[UPCAST2:.+]] = vector.bitcast %[[LOAD2]]
// CHECK: %[[SELECT2:.+]] = arith.select %[[CST1]], %[[INSERT2]], %[[UPCAST2]]
// CHECK: %[[DOWNCAST2:.+]] = vector.bitcast %[[SELECT2]]
// CHECK: vector.store %[[DOWNCAST2]], %[[ALLOC]][%[[INDEX2]]]

// -----

func.func @vector_store_i2_const_index_one_partial_store(%arg0: vector<1xi2>) {
    %0 = memref.alloc() : memref<4x1xi2>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    vector.store %arg0, %0[%c1, %c0] :memref<4x1xi2>, vector<1xi2>
    return
}

// in this test, only emit partial RMW store as the store is within one byte.

// CHECK: func @vector_store_i2_const_index_one_partial_store(
// CHECK-SAME: %[[ARG0:.+]]: vector<1xi2>)
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<1xi8>
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[CST:.+]] = arith.constant dense<[false, true, false, false]>
// CHECK: %[[CST0:.+]] = arith.constant dense<0> : vector<4xi2>
// CHECK: %[[INSERT:.+]] = vector.insert_strided_slice %[[ARG0]], %[[CST0]]
// CHECK-SAME: {offsets = [1], strides = [1]} : vector<1xi2> into vector<4xi2>
// CHECK: %[[LOAD:.+]] = vector.load %[[ALLOC]][%[[C0]]] : memref<1xi8>, vector<1xi8>
// CHECK: %[[UPCAST:.+]] = vector.bitcast %[[LOAD]] : vector<1xi8> to vector<4xi2>
// CHECK: %[[SELECT:.+]] = arith.select %[[CST]], %[[INSERT]], %[[UPCAST]]
// CHECK: %[[DOWNCAST:.+]] = vector.bitcast %[[SELECT]]
// CHECK: vector.store %[[DOWNCAST]], %[[ALLOC]][%[[C0]]]
