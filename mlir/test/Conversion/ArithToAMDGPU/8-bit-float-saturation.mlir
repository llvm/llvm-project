// RUN: mlir-opt --split-input-file %s \
// RUN:   --pass-pipeline='builtin.module(func.func(convert-arith-to-amdgpu{saturate-fp8-truncf=true}))' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @scalar_trunc
// CHECK-SAME: ([[V:%.+]]: f16)
// CHECK-DAG: [[CMin:%.+]] = arith.constant -5.734400e+04 : f16
// CHECK-DAG: [[CMax:%.+]] = arith.constant 5.734400e+04 : f16
// CHECK-DAG: [[CInf:%.+]] = arith.constant 0x7C00 : f16
// CHECK-DAG: [[CNegInf:%.+]] = arith.constant 0xFC00 : f16
// CHECK: [[ISINF:%.+]] = arith.cmpf oeq, [[V]], [[CInf]]
// CHECK: [[ISNEGINF:%.+]] = arith.cmpf oeq, [[V]], [[CNegInf]]
// CHECK: [[ISNAN:%.+]] = arith.cmpf uno, [[V]], [[V]]
// CHECK: [[ISNONFINITE_1:%.+]] = arith.ori [[ISINF]], [[ISNEGINF]]
// CHECK: [[ISNONFINITE:%.+]] = arith.ori [[ISNONFINITE_1]], [[ISNAN]]
// CHECK: [[CLAMPEDBELOW:%.+]] = arith.maximumf [[V]], [[CMin]]
// CHECK: [[CLAMPED:%.+]] = arith.minimumf [[CLAMPEDBELOW]], [[CMax]]
// CHECK: [[SATURATED:%.+]] = arith.select [[ISNONFINITE]], [[V]], [[CLAMPED]]
// CHECK: [[FLOAT:%.+]] = arith.extf [[SATURATED]] : f16 to f32
// CHECK: [[TRUNCV:%.+]] = amdgpu.packed_trunc_2xfp8 [[FLOAT]], undef into undef[word 0] : f32 to vector<4xf8E5M2FNUZ>
// CHECK: [[W:%.+]] = vector.extract [[TRUNCV]][0] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
// CHECK: return [[W]] : f8E5M2FNUZ
func.func @scalar_trunc(%v: f16) -> f8E5M2FNUZ {
  %w = arith.truncf %v : f16 to f8E5M2FNUZ
  return %w : f8E5M2FNUZ
}

// No 0-D test because arith.truncf hasn't been extended to support it.

// -----

// CHECK-LABEL: func.func @vector_trunc
// CHECK-SAME: ([[V:%.+]]: vector<2xf32>) -> vector<2xf8E4M3FNUZ> {
// CHECK-DAG: [[CMin:%.+]] = arith.constant dense<-2.400000e+02> : vector<2xf32>
// CHECK-DAG: [[CMax:%.+]] = arith.constant dense<2.400000e+02> : vector<2xf32>
// CHECK-DAG: [[CInf:%.+]] = arith.constant dense<0x7F800000> : vector<2xf32>
// CHECK-DAG: [[CNegInf:%.+]] = arith.constant dense<0xFF800000> : vector<2xf32>
// CHECK: [[ISINF:%.+]] = arith.cmpf oeq, [[V]], [[CInf]]
// CHECK: [[ISNEGINF:%.+]] = arith.cmpf oeq, [[V]], [[CNegInf]]
// CHECK: [[ISNAN:%.+]] = arith.cmpf uno, [[V]], [[V]]
// CHECK: [[ISNONFINITE_1:%.+]] = arith.ori [[ISINF]], [[ISNEGINF]]
// CHECK: [[ISNONFINITE:%.+]] = arith.ori [[ISNONFINITE_1]], [[ISNAN]]
// CHECK: [[CLAMPEDBELOW:%.+]] = arith.maximumf [[V]], [[CMin]]
// CHECK: [[CLAMPED:%.+]] = arith.minimumf [[CLAMPEDBELOW]], [[CMax]]
// CHECK: [[SATURATED:%.+]] = arith.select [[ISNONFINITE]], [[V]], [[CLAMPED]]
// CHECK: [[F0:%.+]] = vector.extract [[SATURATED]][0]
// CHECK: [[F1:%.+]] = vector.extract [[SATURATED]][1]
// CHECK: [[W0:%.+]] = amdgpu.packed_trunc_2xfp8 [[F0]], [[F1]] into undef[word 0] : f32 to vector<4xf8E4M3FNUZ>
// CHECK: [[W:%.+]] = vector.extract_strided_slice [[W0]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xf8E4M3FNUZ> to vector<2xf8E4M3FNUZ>
// CHECK: return [[W]] : vector<2xf8E4M3FNUZ>
func.func @vector_trunc_short(%v: vector<2xf32>) -> vector<2xf8E4M3FNUZ> {
  %w = arith.truncf %v : vector<2xf32> to vector<2xf8E4M3FNUZ>
  return %w : vector<2xf8E4M3FNUZ>
}
