// RUN: mlir-opt -split-input-file -test-vector-contiguous-extract-strided-slice-to-extract %s | FileCheck %s

// CHECK-LABEL: @extract_strided_slice_to_extract_i8
// CHECK:       vector.extract {{.*}}[0, 0, 0, 0] : vector<8xi8> from vector<8x1x1x2x8xi8>

func.func @extract_strided_slice_to_extract_i8(%arg0 : vector<8x1x1x2x8xi8>) -> vector<8xi8> {
  %1 = vector.extract_strided_slice %arg0 {offsets = [0, 0, 0, 0, 0], sizes = [1, 1, 1, 1, 8], strides = [1, 1, 1, 1, 1]} : vector<8x1x1x2x8xi8> to vector<1x1x1x1x8xi8>
  %2 = vector.shape_cast %1 : vector<1x1x1x1x8xi8> to vector<8xi8>
  return %2 : vector<8xi8>
}

// -----

// CHECK-LABEL: @extract_strided_slice_to_extract_i32
// CHECK:        vector.extract {{.*}}[0, 0, 0, 0, 0] : vector<4xi32> from vector<8x1x2x1x1x4xi32>
func.func @extract_strided_slice_to_extract_i32(%arg0 : vector<8x1x2x1x1x4xi32>) -> vector<4xi32> {
  %1 = vector.extract_strided_slice %arg0 {offsets = [0, 0, 0, 0, 0, 0], sizes = [1, 1, 1, 1, 1, 4], strides = [1, 1, 1, 1, 1, 1]} : vector<8x1x2x1x1x4xi32> to vector<1x1x1x1x1x4xi32>
  %2 = vector.shape_cast %1 : vector<1x1x1x1x1x4xi32> to vector<4xi32>
  return %2 : vector<4xi32>
}

// -----

// CHECK-LABEL: @extract_strided_slice_to_extract_i32_non_contiguous_1
// CHECK:        vector.extract_strided_slice
func.func @extract_strided_slice_to_extract_i32_non_contiguous_1(%arg0 : vector<8x1x2x1x1x4xi32>) -> vector<2xi32> {
  %1 = vector.extract_strided_slice %arg0 {offsets = [0, 0, 0, 0, 0, 0], sizes = [1, 1, 1, 1, 1, 2], strides = [1, 1, 1, 1, 1, 1]} : vector<8x1x2x1x1x4xi32> to vector<1x1x1x1x1x2xi32>
  %2 = vector.shape_cast %1 : vector<1x1x1x1x1x2xi32> to vector<2xi32>
  return %2 : vector<2xi32>
}

// -----

// CHECK-LABEL: @extract_strided_slice_to_extract_i32_non_contiguous_2
// CHECK:        vector.extract_strided_slice
func.func @extract_strided_slice_to_extract_i32_non_contiguous_2(%arg0 : vector<8x1x2x1x1x4xi32>) -> vector<2xi32> {
  %1 = vector.extract_strided_slice %arg0 {offsets = [0, 0, 0, 0, 0, 0], sizes = [1, 1, 2, 1, 1, 1], strides = [1, 1, 1, 1, 1, 1]} : vector<8x1x2x1x1x4xi32> to vector<1x1x2x1x1x1xi32>
  %2 = vector.shape_cast %1 : vector<1x1x2x1x1x1xi32> to vector<2xi32>
  return %2 : vector<2xi32>
}
