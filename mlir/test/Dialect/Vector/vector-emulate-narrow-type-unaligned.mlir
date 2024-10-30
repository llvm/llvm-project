// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=1 memref-load-bitwidth=8" --cse --split-input-file %s | FileCheck %s

func.func @vector_load_i2(%arg1: index, %arg2: index) -> vector<3x3xi2> {
    %0 = memref.alloc() : memref<3x3xi2>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant dense<0> : vector<3x3xi2>
    %1 = vector.load %0[%c2, %c0] : memref<3x3xi2>, vector<3xi2>
    %2 = vector.insert %1, %cst [0] : vector<3xi2> into vector<3x3xi2>
    return %2 : vector<3x3xi2>
}

// CHECK: func @vector_load_i2
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<3xi8>
// CHECK: %[[INDEX:.+]] = arith.constant 1 : index
// CHECK: %[[VEC:.+]] = vector.load %[[ALLOC]][%[[INDEX]]] : memref<3xi8>, vector<2xi8>
// CHECK: %[[VEC_I2:.+]] = vector.bitcast %[[VEC]] : vector<2xi8> to vector<8xi2>
// CHECK: %[[EXCTRACT:.+]] = vector.extract_strided_slice %[[VEC_I2]] {offsets = [2], sizes = [3], strides = [1]} : vector<8xi2> to vector<3xi2>

//-----

func.func @vector_transfer_read_i2() -> vector<3xi2> {
 %0 = memref.alloc() : memref<3x3xi2>
 %c0i2 = arith.constant 0 : i2
 %c0 = arith.constant 0 : index
 %c2 = arith.constant 2 : index
 %1 = vector.transfer_read %0[%c2, %c0], %c0i2 {in_bounds = [true]} : memref<3x3xi2>, vector<3xi2>
 return %1 : vector<3xi2>
}

// CHECK: func @vector_transfer_read_i2
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<3xi8>
// CHECK: %[[INDEX:.+]] = arith.constant 1 : index
// CHECK: %[[READ:.+]] = vector.transfer_read %[[ALLOC]][%[[INDEX]]], %0 : memref<3xi8>, vector<2xi8>
// CHECK: %[[BITCAST:.+]] = vector.bitcast %[[READ]] : vector<2xi8> to vector<8xi2>
// CHECK: vector.extract_strided_slice %[[BITCAST]] {offsets = [2], sizes = [3], strides = [1]} : vector<8xi2> to vector<3xi2>

//-----

func.func @vector_cst_maskedload_i2(%passthru: vector<5xi2>) -> vector<3x5xi2> {
    %0 = memref.alloc() : memref<3x5xi2>
    %cst = arith.constant dense<0> : vector<3x5xi2>
    %mask = vector.constant_mask [3] : vector<5xi1>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %1 = vector.maskedload %0[%c2, %c0], %mask, %passthru :
      memref<3x5xi2>, vector<5xi1>, vector<5xi2> into vector<5xi2>
    %2 = vector.insert %1, %cst [0] : vector<5xi2> into vector<3x5xi2>
    return %2 : vector<3x5xi2>
}

// CHECK: func @vector_cst_maskedload_i2
// CHECK: %[[ORIGINMASK:.+]] = vector.constant_mask [3] : vector<5xi1>
// CHECK: %[[NEWMASK:.+]] = arith.constant dense<true> : vector<2xi1>
// CHECK: %[[VESSEL:.+]] = arith.constant dense<0> : vector<8xi2>
// CHECK: %[[INSERT1:.+]] = vector.insert_strided_slice %arg0, %[[VESSEL]]
// CHECK-SAME: {offsets = [2], strides = [1]} : vector<5xi2> into vector<8xi2>
// CHECK: %[[BITCAST1:.+]] = vector.bitcast %[[INSERT1]] : vector<8xi2> to vector<2xi8>
// CHECK: %[[C2:.+]] = arith.constant 2 : index
// CHECK: %[[MASKEDLOAD:.+]] = vector.maskedload %alloc[%[[C2]]], %[[NEWMASK:.+]], %[[BITCAST1]]
// CHECK-SAME: : memref<4xi8>, vector<2xi1>, vector<2xi8> into vector<2xi8>
// CHECK: %[[BITCAST2:.+]] = vector.bitcast %[[MASKEDLOAD]] : vector<2xi8> to vector<8xi2>
// CHECK: %[[CST2:.+]] = arith.constant dense<false> : vector<8xi1>
// CHECK: %[[INSERT2:.+]] = vector.insert_strided_slice %[[ORIGINMASK]], %[[CST2]]
// CHECK-SAME: {offsets = [2], strides = [1]} : vector<5xi1> into vector<8xi1>
// CHECK: %[[SELECT:.+]] = arith.select %[[INSERT2]], %[[BITCAST2]], %[[INSERT1]] : vector<8xi1>, vector<8xi2>
// CHECK: vector.extract_strided_slice %[[SELECT]] {offsets = [2], sizes = [5], strides = [1]} : vector<8xi2> to vector<5xi2> 
