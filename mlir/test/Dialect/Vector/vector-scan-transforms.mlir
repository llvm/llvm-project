// RUN: mlir-opt %s --test-vector-scan-lowering | FileCheck %s

// CHECK-LABEL: func @scan1d_inc
// CHECK-SAME: %[[ARG0:.*]]: vector<2xi32>,
// CHECK-SAME: %[[ARG1:.*]]: vector<i32>
// CHECK:      %[[A:.*]] = arith.constant dense<0> : vector<2xi32>
// CHECK:      %[[B:.*]] = vector.extract_strided_slice %[[ARG0]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xi32> to vector<1xi32>
// CHECK:      %[[C:.*]] = vector.insert_strided_slice %[[B]], %[[A]] {offsets = [0], strides = [1]} : vector<1xi32> into vector<2xi32>
// CHECK:      %[[D:.*]] = vector.extract_strided_slice %[[ARG0]] {offsets = [1], sizes = [1], strides = [1]} : vector<2xi32> to vector<1xi32>
// CHECK:      %[[E:.*]] = arith.addi %[[B]], %[[D]] : vector<1xi32>
// CHECK:      %[[F:.*]] = vector.insert_strided_slice %[[E]], %[[C]] {offsets = [1], strides = [1]} : vector<1xi32> into vector<2xi32>
// CHECK:      %[[G:.*]] = vector.extract %[[E]][0] : i32 from vector<1xi32>
// CHECK:      %[[H:.*]] = vector.broadcast %[[G]] : i32 to vector<i32>
// CHECK:      return %[[F]], %[[H]] : vector<2xi32>, vector<i32>
func.func @scan1d_inc(%arg0 : vector<2xi32>, %arg1 : vector<i32>) -> (vector<2xi32>, vector<i32>) {
  %0:2 = vector.scan <add>, %arg0, %arg1 {inclusive = true, reduction_dim = 0} :
    vector<2xi32>, vector<i32>
  return %0#0, %0#1 : vector<2xi32>, vector<i32>
}

// CHECK-LABEL: func @scan1d_exc
// CHECK-SAME: %[[ARG0:.*]]: vector<2xi32>,
// CHECK-SAME: %[[ARG1:.*]]: vector<i32>
// CHECK:      %[[A:.*]] = arith.constant dense<0> : vector<2xi32>
// CHECK:      %[[B:.*]] = vector.extract_strided_slice %[[ARG0]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xi32> to vector<1xi32>
// CHECK:      %[[C:.*]] = vector.broadcast %[[ARG1]] : vector<i32> to vector<1xi32>
// CHECK:      %[[D:.*]] = vector.insert_strided_slice %[[C]], %[[A]] {offsets = [0], strides = [1]} : vector<1xi32> into vector<2xi32>
// CHECK:      %[[E:.*]] = arith.addi %[[C]], %[[B]] : vector<1xi32>
// CHECK:      %[[F:.*]] = vector.insert_strided_slice %[[E]], %[[D]] {offsets = [1], strides = [1]} : vector<1xi32> into vector<2xi32>
// CHECK:      %[[G:.*]] = vector.extract %[[E]][0] : i32 from vector<1xi32>
// CHECK:      %[[H:.*]] = vector.broadcast %[[G]] : i32 to vector<i32>
// CHECK:      return %[[F]], %[[H]] : vector<2xi32>, vector<i32>
func.func @scan1d_exc(%arg0 : vector<2xi32>, %arg1 : vector<i32>) -> (vector<2xi32>, vector<i32>) {
  %0:2 = vector.scan <add>, %arg0, %arg1 {inclusive = false, reduction_dim = 0} :
    vector<2xi32>, vector<i32>
  return %0#0, %0#1 : vector<2xi32>, vector<i32>
}

// CHECK-LABEL: func @scan2d_mul_dim0
// CHECK-SAME: %[[ARG0:.*]]: vector<2x3xi32>,
// CHECK-SAME: %[[ARG1:.*]]: vector<3xi32>
// CHECK:      %[[A:.*]] = arith.constant dense<0> : vector<2x3xi32>
// CHECK:      %[[B:.*]] = vector.extract_strided_slice %[[ARG0]] {offsets = [0, 0], sizes = [1, 3], strides = [1, 1]} : vector<2x3xi32> to vector<1x3xi32>
// CHECK:      %[[C:.*]] = vector.insert_strided_slice %[[B]], %[[A]] {offsets = [0, 0], strides = [1, 1]} : vector<1x3xi32> into vector<2x3xi32>
// CHECK:      %[[D:.*]] = vector.extract_strided_slice %[[ARG0]] {offsets = [1, 0], sizes = [1, 3], strides = [1, 1]} : vector<2x3xi32> to vector<1x3xi32>
// CHECK:      %[[E:.*]] = arith.muli %[[B]], %[[D]] : vector<1x3xi32>
// CHECK:      %[[F:.*]] = vector.insert_strided_slice %[[E]], %[[C]] {offsets = [1, 0], strides = [1, 1]} : vector<1x3xi32> into vector<2x3xi32>
// CHECK:      %[[G:.*]] = vector.shape_cast %[[E]] : vector<1x3xi32> to vector<3xi32>
// CHECK:      return %[[F]], %[[G]] : vector<2x3xi32>, vector<3xi32>
func.func @scan2d_mul_dim0(%arg0 : vector<2x3xi32>, %arg1 : vector<3xi32>) -> (vector<2x3xi32>, vector<3xi32>) {
  %0:2 = vector.scan <mul>, %arg0, %arg1 {inclusive = true, reduction_dim = 0} :
    vector<2x3xi32>, vector<3xi32>
  return %0#0, %0#1 : vector<2x3xi32>, vector<3xi32>
}

// CHECK-LABEL: func @scan2d_mul_dim1
// CHECK-SAME: %[[ARG0:.*]]: vector<2x3xi32>,
// CHECK-SAME: %[[ARG1:.*]]: vector<2xi32>
// CHECK:      %[[A:.*]] = arith.constant dense<0> : vector<2x3xi32>
// CHECK:      %[[B:.*]] = vector.extract_strided_slice %[[ARG0]] {offsets = [0, 0], sizes = [2, 1], strides = [1, 1]} : vector<2x3xi32> to vector<2x1xi32>
// CHECK:      %[[C:.*]] = vector.insert_strided_slice %[[B]], %[[A]] {offsets = [0, 0], strides = [1, 1]} : vector<2x1xi32> into vector<2x3xi32>
// CHECK:      %[[D:.*]] = vector.extract_strided_slice %[[ARG0]] {offsets = [0, 1], sizes = [2, 1], strides = [1, 1]} : vector<2x3xi32> to vector<2x1xi32>
// CHECK:      %[[E:.*]] = arith.muli %[[B]], %[[D]] : vector<2x1xi32>
// CHECK:      %[[F:.*]] = vector.insert_strided_slice %[[E]], %[[C]] {offsets = [0, 1], strides = [1, 1]} : vector<2x1xi32> into vector<2x3xi32>
// CHECK:      %[[G:.*]] = vector.extract_strided_slice %[[ARG0]] {offsets = [0, 2], sizes = [2, 1], strides = [1, 1]} : vector<2x3xi32> to vector<2x1xi32>
// CHECK:      %[[H:.*]] = arith.muli %[[E]], %[[G]] : vector<2x1xi32>
// CHECK:      %[[I:.*]] = vector.insert_strided_slice %[[H]], %[[F]] {offsets = [0, 2], strides = [1, 1]} : vector<2x1xi32> into vector<2x3xi32>
// CHECK:      %[[J:.*]] = vector.shape_cast %[[H]] : vector<2x1xi32> to vector<2xi32>
// CHECK:      return %[[I]], %[[J]] : vector<2x3xi32>, vector<2xi32>
func.func @scan2d_mul_dim1(%arg0 : vector<2x3xi32>, %arg1 : vector<2xi32>) -> (vector<2x3xi32>, vector<2xi32>) {
  %0:2 = vector.scan <mul>, %arg0, %arg1 {inclusive = true, reduction_dim = 1} :
    vector<2x3xi32>, vector<2xi32>
  return %0#0, %0#1 : vector<2x3xi32>, vector<2xi32>
}

// CHECK-LABEL: func @scan3d_mul_dim1
// CHECK-SAME: %[[ARG0:.*]]: vector<4x2x3xf32>,
// CHECK-SAME: %[[ARG1:.*]]: vector<4x3xf32>
// CHECK:      %[[A:.*]] = arith.constant dense<0.000000e+00> : vector<4x2x3xf32>
// CHECK:      %[[B:.*]] = vector.extract_strided_slice %[[ARG0]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xf32> to vector<4x1x3xf32>
// CHECK:      %[[C:.*]] = vector.shape_cast %[[ARG1]] : vector<4x3xf32> to vector<4x1x3xf32>
// CHECK:      %[[D:.*]] = vector.insert_strided_slice %[[C]], %[[A]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<4x1x3xf32> into vector<4x2x3xf32>
// CHECK:      %[[E:.*]] = arith.mulf %[[C]], %[[B]] : vector<4x1x3xf32>
// CHECK:      %[[F:.*]] = vector.insert_strided_slice %[[E]], %[[D]] {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<4x1x3xf32> into vector<4x2x3xf32>
// CHECK:      %[[G:.*]] = vector.shape_cast %[[E]] : vector<4x1x3xf32> to vector<4x3xf32>
// CHECK:      return %[[F]], %[[G]] : vector<4x2x3xf32>, vector<4x3xf32>
func.func @scan3d_mul_dim1(%arg0 : vector<4x2x3xf32>, %arg1 : vector<4x3xf32>) -> (vector<4x2x3xf32>, vector<4x3xf32>) {
  %0:2 = vector.scan <mul>, %arg0, %arg1 {inclusive = false, reduction_dim = 1} :
    vector<4x2x3xf32>, vector<4x3xf32>
  return %0#0, %0#1 : vector<4x2x3xf32>, vector<4x3xf32>
}
