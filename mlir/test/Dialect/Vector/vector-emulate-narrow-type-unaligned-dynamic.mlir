// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=1 memref-load-bitwidth=8" --cse --split-input-file %s | FileCheck %s

// CHECK: #map = affine_map<()[s0, s1] -> ((s0 * 3 + s1) floordiv 4)>
// CHECK: #map1 = affine_map<()[s0, s1] -> ((s0 * 3 + s1) mod 4)>
func.func @vector_load_i2(%arg1: index, %arg2: index) -> vector<3x3xi2> {
    %0 = memref.alloc() : memref<3x3xi2>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant dense<0> : vector<3x3xi2>
    %1 = vector.load %0[%arg1, %arg2] : memref<3x3xi2>, vector<3xi2>
    %2 = vector.insert %1, %cst [0] : vector<3xi2> into vector<3x3xi2>
    return %2 : vector<3x3xi2>
}

// CHECK: func @vector_load_i2
// CHECK: %[[ALLOC:.+]]= memref.alloc() : memref<3xi8>
// CHECK: %[[LOADADDR1:.+]] = affine.apply #map()[%arg0, %arg1]
// CHECK: %[[LOADADDR2:.+]] = affine.apply #map1()[%arg0, %arg1]
// CHECK: %[[EMULATED_LOAD:.+]] = vector.load %alloc[%[[LOADADDR1]]] : memref<3xi8>, vector<2xi8>
// CHECK: %[[BITCAST:.+]] = vector.bitcast %[[EMULATED_LOAD]] : vector<2xi8> to vector<8xi2>
// CHECK: %[[ZERO:.+]] = arith.constant dense<0> : vector<3xi2>
// CHECK: %[[EXTRACT:.+]] = vector.extract %[[BITCAST]][%[[LOADADDR2]]] : i2 from vector<8xi2>
// CHECK: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[OFFSET:.+]] = arith.addi %[[LOADADDR2]], %[[C1]] : index
// CHECK: %[[EXTRACT2:.+]] = vector.extract %[[BITCAST]][%[[OFFSET]]] : i2 from vector<8xi2>
// CHECK: %[[C2:.+]] = arith.constant 2 : index
// CHECK: %[[OFFSET2:.+]] = arith.addi %1, %c2 : index
// CHECK: %[[EXTRACT3:.+]] = vector.extract %[[BITCAST]][%[[OFFSET2]]] : i2 from vector<8xi2>

//-----

func.func @vector_transfer_read_i2(%arg1: index, %arg2: index) -> vector<3xi2> {
 %0 = memref.alloc() : memref<3x3xi2>
 %c0i2 = arith.constant 0 : i2
 %1 = vector.transfer_read %0[%arg1, %arg2], %c0i2 {in_bounds = [true]} : memref<3x3xi2>, vector<3xi2>
 return %1 : vector<3xi2>
}

// CHECK: func @vector_transfer_read_i2
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<3xi8>
// CHECK: %[[C0:.+]] = arith.extui %c0_i2 : i2 to i8
// CHECK: %[[LOADADDR1:.+]] = affine.apply #map()[%arg0, %arg1]
// CHECK: %[[LOADADDR2:.+]] = affine.apply #map1()[%arg0, %arg1]
// CHECK: %[[READ:.+]] = vector.transfer_read %[[ALLOC]][%[[LOADADDR1]]], %[[C0]] : memref<3xi8>, vector<2xi8>
// CHECK: %[[BITCAST:.+]] = vector.bitcast %[[READ]] : vector<2xi8> to vector<8xi2>
// CHECK: %[[CST:.+]] = arith.constant dense<0> : vector<3xi2>
// CHECK: %[[EXTRACT:.+]] = vector.extract %[[BITCAST]][%[[LOADADDR2]]] : i2 from vector<8xi2>
// CHECK: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[ADDI:.+]] = arith.addi %[[LOADADDR2]], %[[C1]] : index
// CHECK: %[[EXTRACT2:.+]] = vector.extract %[[BITCAST]][%[[ADDI]]] : i2 from vector<8xi2>
// CHECK: %[[C2:.+]] = arith.constant 2 : index
// CHECK: %[[ADDI2:.+]] = arith.addi %[[LOADADDR2]], %[[C2]] : index
// CHECK: %[[EXTRACT3:.+]] = vector.extract %[[BITCAST]][%[[ADDI2]]] : i2 from vector<8xi2>
