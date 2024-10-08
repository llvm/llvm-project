// RUN: mlir-opt --test-vector-contiguous-extract-strided-slice-to-extract %s | FileCheck %s

// CHECK-LABEL: @contiguous
// CHECK:        %[[EXTRACT:.+]] = vector.extract {{.*}}[0, 0, 0, 0, 0] : vector<4xi32> from vector<8x1x2x1x1x4xi32>
// CHECK:       return %[[EXTRACT]] :  vector<4xi32>
func.func @contiguous(%arg0 : vector<8x1x2x1x1x4xi32>) -> vector<4xi32> {
  %1 = vector.extract_strided_slice %arg0 {offsets = [0, 0, 0, 0, 0, 0], sizes = [1, 1, 1, 1, 1, 4], strides = [1, 1, 1, 1, 1, 1]} : vector<8x1x2x1x1x4xi32> to vector<1x1x1x1x1x4xi32>
  %2 = vector.shape_cast %1 : vector<1x1x1x1x1x4xi32> to vector<4xi32>
  return %2 : vector<4xi32>
}

// CHECK-LABEL: @non_full_size
// CHECK:        vector.extract_strided_slice
func.func @non_full_size(%arg0 : vector<8x1x2x1x1x4xi32>) -> vector<2xi32> {
  %1 = vector.extract_strided_slice %arg0 {offsets = [0, 0, 0, 0, 0, 0], sizes = [1, 1, 1, 1, 1, 2], strides = [1, 1, 1, 1, 1, 1]} : vector<8x1x2x1x1x4xi32> to vector<1x1x1x1x1x2xi32>
  %2 = vector.shape_cast %1 : vector<1x1x1x1x1x2xi32> to vector<2xi32>
  return %2 : vector<2xi32>
}

// CHECK-LABEL: @non_contiguous
// CHECK:        vector.extract_strided_slice
func.func @non_full_inner_size(%arg0 : vector<8x1x2x1x1x4xi32>) -> vector<2xi32> {
  %1 = vector.extract_strided_slice %arg0 {offsets = [0, 0, 0, 0, 0, 0], sizes = [1, 1, 2, 1, 1, 1], strides = [1, 1, 1, 1, 1, 1]} : vector<8x1x2x1x1x4xi32> to vector<1x1x2x1x1x1xi32>
  %2 = vector.shape_cast %1 : vector<1x1x2x1x1x1xi32> to vector<2xi32>
  return %2 : vector<2xi32>
}
