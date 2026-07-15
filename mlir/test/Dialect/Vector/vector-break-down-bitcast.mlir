// RUN: mlir-opt -split-input-file -test-vector-break-down-bitcast %s | FileCheck %s

// CHECK-LABEL: func.func @bitcast_f16_to_f32
//  CHECK-SAME: (%[[INPUT:.+]]: vector<8xf16>)
func.func @bitcast_f16_to_f32(%input: vector<8xf16>) -> vector<4xf32> {
  %0 = vector.bitcast %input : vector<8xf16> to vector<4xf32>
  return %0: vector<4xf32>
}

// CHECK: %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK: %[[EXTRACT0:.+]] = vector.extract_strided_slice %[[INPUT]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK: %[[CAST0:.+]] = vector.bitcast %[[EXTRACT0]] : vector<4xf16> to vector<2xf32>
// CHECK: %[[INSERT0:.+]] = vector.insert_strided_slice %[[CAST0]], %[[INIT]] {offsets = [0], strides = [1]} : vector<2xf32> into vector<4xf32>
// CHECK: %[[EXTRACT1:.+]] = vector.extract_strided_slice %[[INPUT]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK: %[[CAST1:.+]] = vector.bitcast %[[EXTRACT1]] : vector<4xf16> to vector<2xf32>
// CHECK: %[[INSERT1:.+]] = vector.insert_strided_slice %[[CAST1]], %[[INSERT0]] {offsets = [2], strides = [1]} : vector<2xf32> into vector<4xf32>
// CHECK: return %[[INSERT1]]

// -----

// CHECK-LABEL: func.func @bitcast_i8_to_i32
//  CHECK-SAME: (%[[INPUT:.+]]: vector<16xi8>)
func.func @bitcast_i8_to_i32(%input: vector<16xi8>) -> vector<4xi32> {
  %0 = vector.bitcast %input : vector<16xi8> to vector<4xi32>
  return %0: vector<4xi32>
}

// CHECK: %[[INIT:.+]] = arith.constant dense<0> : vector<4xi32>
// CHECK: %[[EXTRACT0:.+]] = vector.extract_strided_slice %[[INPUT]] {offsets = [0], sizes = [4], strides = [1]} : vector<16xi8> to vector<4xi8>
// CHECK: %[[CAST0:.+]] = vector.bitcast %[[EXTRACT0]] : vector<4xi8> to vector<1xi32>
// CHECK: %[[INSERT0:.+]] = vector.insert_strided_slice %[[CAST0]], %[[INIT]] {offsets = [0], strides = [1]} : vector<1xi32> into vector<4xi32>
// CHECK: %[[EXTRACT1:.+]] = vector.extract_strided_slice %[[INPUT]] {offsets = [4], sizes = [4], strides = [1]} : vector<16xi8> to vector<4xi8>
// CHECK: %[[CAST1:.+]] = vector.bitcast %[[EXTRACT1]] : vector<4xi8> to vector<1xi32>
// CHECK: %[[INSERT1:.+]] = vector.insert_strided_slice %[[CAST1]], %[[INSERT0]] {offsets = [1], strides = [1]} : vector<1xi32> into vector<4xi32>
// CHECK: %[[EXTRACT2:.+]] = vector.extract_strided_slice %[[INPUT]] {offsets = [8], sizes = [4], strides = [1]} : vector<16xi8> to vector<4xi8>
// CHECK: %[[CAST2:.+]] = vector.bitcast %[[EXTRACT2]] : vector<4xi8> to vector<1xi32>
// CHECK: %[[INSERT2:.+]] = vector.insert_strided_slice %[[CAST2]], %[[INSERT1]] {offsets = [2], strides = [1]} : vector<1xi32> into vector<4xi32>
// CHECK: %[[EXTRACT3:.+]] = vector.extract_strided_slice %[[INPUT]] {offsets = [12], sizes = [4], strides = [1]} : vector<16xi8> to vector<4xi8>
// CHECK: %[[CAST3:.+]] = vector.bitcast %[[EXTRACT3]] : vector<4xi8> to vector<1xi32>
// CHECK: %[[INSERT3:.+]] = vector.insert_strided_slice %[[CAST3]], %[[INSERT2]] {offsets = [3], strides = [1]} : vector<1xi32> into vector<4xi32>
// CHECK: return %[[INSERT3]]

// -----

// Scalable vectors are not supported!

// CHECK-LABEL: func.func @bitcast_scalable_negative
// CHECK: vector.bitcast
func.func @bitcast_scalable_negative(%input: vector<[8]xf16>) -> vector<[4]xf32> {
  %0 = vector.bitcast %input : vector<[8]xf16> to vector<[4]xf32>
  return %0: vector<[4]xf32>
}
