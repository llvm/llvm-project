// RUN: mlir-opt -split-input-file -test-vector-extract-strided-slice-lowering %s | FileCheck %s

// CHECK-LABEL: func.func @extract_strided_slice_1D
//  CHECK-SAME: (%[[INPUT:.+]]: vector<8xf16>)
func.func @extract_strided_slice_1D(%input: vector<8xf16>) -> vector<4xf16> {
  %0 = vector.extract_strided_slice %input {offsets = [1], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
  return %0: vector<4xf16>
}

// CHECK: %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<4xf16>
// CHECK: %[[E0:.+]] = vector.extract %[[INPUT]][1] : vector<8xf16>
// CHECK: %[[E1:.+]] = vector.extract %[[INPUT]][2] : vector<8xf16>
// CHECK: %[[E2:.+]] = vector.extract %[[INPUT]][3] : vector<8xf16>
// CHECK: %[[E3:.+]] = vector.extract %[[INPUT]][4] : vector<8xf16>
// CHECK: %[[I0:.+]] = vector.insert %[[E0]], %[[INIT]] [0] : f16 into vector<4xf16>
// CHECK: %[[I1:.+]] = vector.insert %[[E1]], %[[I0]] [1] : f16 into vector<4xf16>
// CHECK: %[[I2:.+]] = vector.insert %[[E2]], %[[I1]] [2] : f16 into vector<4xf16>
// CHECK: %[[I3:.+]] = vector.insert %[[E3]], %[[I2]] [3] : f16 into vector<4xf16>
// CHECK: return %[[I3]]


// -----

// CHECK-LABEL: func.func @extract_strided_slice_2D
func.func @extract_strided_slice_2D(%input: vector<1x8xf16>) -> vector<1x4xf16> {
  // CHECK: vector.extract_strided_slice
  %0 = vector.extract_strided_slice %input {offsets = [0, 1], sizes = [1, 4], strides = [1, 1]} : vector<1x8xf16> to vector<1x4xf16>
  return %0: vector<1x4xf16>
}
