// RUN: mlir-opt %s -test-vector-contraction-conversion=vector-flat-transpose=1 | FileCheck %s

// Tests for lowering 2-D vector.transpose into vector.flat_transpose.
//
// TODO(ajcbik,ntv): having ShapeCastOp2DDownCastRewritePattern and
//                   ShapeCastOp2DUpCastRewritePattern too early in
//                   the greedy rewriting patterns misses opportunities
//                   to fold shape casts!

// No shape cast folding expected.
//
// CHECK-LABEL: func @transpose44_44(
// CHECK-SAME:  %[[A:.*]]: vector<4x4xf32>
// CHECK:       %[[T0:.*]] = vector.extract %[[A]][0] : vector<4x4xf32>
// CHECK:       %[[T8:.*]] = vector.flat_transpose %{{.*}} {columns = 4 : i32, rows = 4 : i32} : vector<16xf32> -> vector<16xf32>
// CHECK:       %[[T9:.*]] = vector.extract_strided_slice %[[T8]] {offsets = [0], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
//
func @transpose44_44(%arg0: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.transpose %arg0, [1, 0] : vector<4x4xf32> to vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// Folds preceding shape cast as expected,
// no following shape cast folding expected.
//
// CHECK-LABEL: func @transpose16_44(
// CHECK-SAME:  %[[A:.*]]: vector<16xf32>
// CHECK:       %[[T0:.*]] = vector.flat_transpose %[[A]] {columns = 4 : i32, rows = 4 : i32} : vector<16xf32> -> vector<16xf32>
// CHECK:       %[[T1:.*]] = vector.extract_strided_slice %[[T0]] {offsets = [0], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
//
func @transpose16_44(%arg0: vector<16xf32>) -> vector<4x4xf32> {
  %0 = vector.shape_cast %arg0 : vector<16xf32> to vector<4x4xf32>
  %1 = vector.transpose %0, [1, 0] : vector<4x4xf32> to vector<4x4xf32>
  return %1 : vector<4x4xf32>
}

// No preceding shape cast folding expected,
// but FAILS to fold following cast.
//
// CHECK-LABEL: func @transpose44_16(
// CHECK-SAME:  %[[A:.*]]: vector<4x4xf32>
// CHECK:       %[[T0:.*]] = vector.extract %[[A]][0] : vector<4x4xf32>
// CHECK:       %[[T8:.*]] = vector.flat_transpose %{{.*}} {columns = 4 : i32, rows = 4 : i32} : vector<16xf32> -> vector<16xf32>
func @transpose44_16(%arg0: vector<4x4xf32>) -> vector<16xf32> {
  %0 = vector.transpose %arg0, [1, 0] : vector<4x4xf32> to vector<4x4xf32>
  %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
  return %1 : vector<16xf32>
}

// Folds preceding shape cast as expected,
// but FAILS to fold following cast.
//
// CHECK-LABEL: func @transpose16_16(
// CHECK-SAME:  %[[A:.*]]: vector<16xf32>
// CHECK:       %[[T0:.*]] = vector.flat_transpose %[[A]] {columns = 4 : i32, rows = 4 : i32} : vector<16xf32> -> vector<16xf32>
//
func @transpose16_16(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = vector.shape_cast %arg0 : vector<16xf32> to vector<4x4xf32>
  %1 = vector.transpose %0, [1, 0] : vector<4x4xf32> to vector<4x4xf32>
  %2 = vector.shape_cast %1 : vector<4x4xf32> to vector<16xf32>
  return %2 : vector<16xf32>
}
