// RUN: mlir-opt %s -transform-interpreter | FileCheck %s

// Check the AVX-based approximation of math.rsqrt.

// We only approximate rsqrt for vectors.
// CHECK-LABEL:    func @rsqrt_scalar
// CHECK:            math.rsqrt
func.func @rsqrt_scalar(%arg0: f32) -> f32 {
  %0 = math.rsqrt %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:    func @rsqrt_vector_8xf32(
// CHECK-SAME:       %[[VAL_0:.*]]: vector<8xf32>) -> vector<8xf32> {
// CHECK:   %[[VAL_1:.*]] = arith.constant dense<0x7F800000> : vector<8xf32>
// CHECK:   %[[VAL_2:.*]] = arith.constant dense<1.500000e+00> : vector<8xf32>
// CHECK:   %[[VAL_3:.*]] = arith.constant dense<-5.000000e-01> : vector<8xf32>
// CHECK:   %[[VAL_4:.*]] = arith.constant dense<1.17549435E-38> : vector<8xf32>
// CHECK:   %[[VAL_5:.*]] = arith.mulf %[[VAL_0]], %[[VAL_3]] : vector<8xf32>
// CHECK:   %[[VAL_6:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_4]] : vector<8xf32>
// CHECK:   %[[VAL_7:.*]] = arith.cmpf oeq, %[[VAL_0]], %[[VAL_1]] : vector<8xf32>
// CHECK:   %[[VAL_8:.*]] = arith.ori %[[VAL_6]], %[[VAL_7]] : vector<8xi1>
// CHECK:   %[[VAL_9:.*]] = x86.avx.rsqrt %[[VAL_0]] : vector<8xf32>
// CHECK:   %[[VAL_10:.*]] = arith.mulf %[[VAL_5]], %[[VAL_9]] : vector<8xf32>
// CHECK:   %[[VAL_11:.*]] = math.fma %[[VAL_9]], %[[VAL_10]], %[[VAL_2]] : vector<8xf32>
// CHECK:   %[[VAL_12:.*]] = arith.mulf %[[VAL_9]], %[[VAL_11]] : vector<8xf32>
// CHECK:   %[[VAL_13:.*]] = arith.select %[[VAL_8]], %[[VAL_9]], %[[VAL_12]] : vector<8xi1>, vector<8xf32>
// CHECK:   return %[[VAL_13]] : vector<8xf32>
// CHECK: }
func.func @rsqrt_vector_8xf32(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.rsqrt %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

// Virtual vector width is not a multiple of an AVX2 vector width.
//
// CHECK-LABEL:   func @rsqrt_vector_5xf32
// CHECK:           math.rsqrt
func.func @rsqrt_vector_5xf32(%arg0: vector<5xf32>) -> vector<5xf32> {
  %0 = math.rsqrt %arg0 : vector<5xf32>
  return %0 : vector<5xf32>
}

// One dimensional virtual vector expanded and unrolled into multiple AVX2-sized
// vectors.
//
// CHECK-LABEL:  func @rsqrt_vector_16xf32(
// CHECK-SAME:     %[[ARG:.*]]: vector<16xf32>
// CHECK-SAME:   ) -> vector<16xf32>
// CHECK:          %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<2x8xf32>
// CHECK:          %[[EXPAND:.*]] = vector.shape_cast %[[ARG]] : vector<16xf32> to vector<2x8xf32>
// CHECK:          %[[VEC0:.*]] = vector.extract %[[EXPAND]][0]
// CHECK:          %[[RSQRT0:.*]] = x86.avx.rsqrt %[[VEC0]]
// CHECK:          %[[VEC1:.*]] = vector.extract %[[EXPAND]][1]
// CHECK:          %[[RSQRT1:.*]] = x86.avx.rsqrt %[[VEC1]]
// CHECK:          %[[RESULT0:.*]] = vector.insert %[[RSQRT0]], %[[INIT]] [0]
// CHECK:          %[[RESULT1:.*]] = vector.insert %[[RSQRT1]], %[[RESULT0]] [1]
// CHECK:          %[[RSQRT:.*]] = vector.shape_cast %[[RESULT1]] : vector<2x8xf32> to vector<16xf32>
func.func @rsqrt_vector_16xf32(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = math.rsqrt %arg0 : vector<16xf32>
  return %0 : vector<16xf32>
}

// Two dimensional virtual vector unrolled into multiple AVX2-sized vectors.
//
// CHECK-LABEL:  func @rsqrt_vector_2x8xf32(
// CHECK-SAME:     %[[ARG:.*]]: vector<2x8xf32>
// CHECK-SAME:   ) -> vector<2x8xf32>
// CHECK:          %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<2x8xf32>
// CHECK-NOT:      vector.shape_cast
// CHECK:          %[[VEC0:.*]] = vector.extract %[[ARG]][0]
// CHECK:          %[[RSQRT0:.*]] = x86.avx.rsqrt %[[VEC0]]
// CHECK:          %[[VEC1:.*]] = vector.extract %[[ARG]][1]
// CHECK:          %[[RSQRT1:.*]] = x86.avx.rsqrt %[[VEC1]]
// CHECK:          %[[RESULT0:.*]] = vector.insert %[[RSQRT0]], %[[INIT]] [0]
// CHECK:          %[[RESULT1:.*]] = vector.insert %[[RSQRT1]], %[[RESULT0]] [1]
// CHECK-NOT:      vector.shape_cast
func.func @rsqrt_vector_2x8xf32(%arg0: vector<2x8xf32>) -> vector<2x8xf32> {
  %0 = math.rsqrt %arg0 : vector<2x8xf32>
  return %0 : vector<2x8xf32>
}

// Two dimensional virtual vector expanded and unrolled into multiple AVX2-sized
// vectors.
//
// CHECK-LABEL:  func @rsqrt_vector_2x16xf32(
// CHECK-SAME:     %[[ARG:.*]]: vector<2x16xf32>
// CHECK-SAME:   ) -> vector<2x16xf32>
// CHECK:          %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<2x2x8xf32>
// CHECK:          %[[EXPAND:.*]] = vector.shape_cast %[[ARG]] : vector<2x16xf32> to vector<2x2x8xf32>
// CHECK:          %[[VEC00:.*]] = vector.extract %[[EXPAND]][0, 0]
// CHECK:          %[[RSQRT00:.*]] = x86.avx.rsqrt %[[VEC00]]
// CHECK:          %[[VEC01:.*]] = vector.extract %[[EXPAND]][0, 1]
// CHECK:          %[[RSQRT01:.*]] = x86.avx.rsqrt %[[VEC01]]
// CHECK:          %[[VEC10:.*]] = vector.extract %[[EXPAND]][1, 0]
// CHECK:          %[[RSQRT10:.*]] = x86.avx.rsqrt %[[VEC10]]
// CHECK:          %[[VEC11:.*]] = vector.extract %[[EXPAND]][1, 1]
// CHECK:          %[[RSQRT11:.*]] = x86.avx.rsqrt %[[VEC11]]
// CHECK:          %[[RESULT0:.*]] = vector.insert %[[RSQRT00]], %[[INIT]] [0, 0]
// CHECK:          %[[RESULT1:.*]] = vector.insert %[[RSQRT01]], %[[RESULT0]] [0, 1]
// CHECK:          %[[RESULT2:.*]] = vector.insert %[[RSQRT10]], %[[RESULT1]] [1, 0]
// CHECK:          %[[RESULT3:.*]] = vector.insert %[[RSQRT11]], %[[RESULT2]] [1, 1]
// CHECK:          %[[RSQRT:.*]] = vector.shape_cast %[[RESULT3]] : vector<2x2x8xf32> to vector<2x16xf32>
func.func @rsqrt_vector_2x16xf32(%arg0: vector<2x16xf32>) -> vector<2x16xf32> {
  %0 = math.rsqrt %arg0 : vector<2x16xf32>
  return %0 : vector<2x16xf32>
}

// CHECK-LABEL: @rsqrt_f16
func.func @rsqrt_f16(%arg0 : vector<2x8xf16>) -> vector<2x8xf16> {
  // CHECK-NOT: math.rsqrt
  %0 = "math.rsqrt"(%arg0) : (vector<2x8xf16>) -> vector<2x8xf16>
  return %0 : vector<2x8xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.math.f32_expansion
      transform.apply_patterns.x86.math_rsqrt_approximation
    } : !transform.any_op
    transform.yield
  }
}
