// RUN: mlir-opt %s -split-input-file -test-rank-reduce-strided-slice-ops -verify-diagnostics | FileCheck %s


// **---------------------------------------------**
//       Tests of vector.extract_strided_slice
// **---------------------------------------------**


// The 6 elements extracted are contiguous, so this can be expressed as a rank-1 vector.extract_strided_slice.

// CHECK-LABEL: @extract_strided_slice_2D_to_1D(
//  CHECK-SAME:    %[[A:.*]]: vector<5x2xi8>) -> vector<3x2xi8> {
//       CHECK: %[[SC:.*]] = vector.shape_cast %[[A]] : vector<5x2xi8> to vector<10xi8>
//       CHECK: %[[EXTRACTED:.*]] = vector.extract_strided_slice %[[SC]]
//  CHECK-SAME:    {offsets = [2], sizes = [6], strides = [1]} : vector<10xi8> to vector<6xi8>
//       CHECK: %[[CASTED:.*]] = vector.shape_cast %[[EXTRACTED]] : vector<6xi8> to vector<3x2xi8>
//       CHECK: return %[[CASTED]] : vector<3x2xi8>
func.func @extract_strided_slice_2D_to_1D(%arg0 : vector<5x2xi8>) -> vector<3x2xi8> {
  %extracted = vector.extract_strided_slice %arg0 {offsets = [1, 0], sizes = [3, 2], strides = [1, 1]} : vector<5x2xi8> to vector<3x2xi8>
  return %extracted : vector<3x2xi8>
}

// -----

// The 5 elements extracted are not contiguous, so this cannot be expressed as a rank-1 vector.extract_strided_slice.

// CHECK-LABEL: @negative_extract_strided_slice_2D_to_1D(
//  CHECK-SAME:    %[[A:.*]]: vector<5x2xi8>) -> vector<5x1xi8> {
//       CHECK: %[[EXTRACTED:.*]] = vector.extract_strided_slice %[[A]]
//       CHECK: return %[[EXTRACTED]] : vector<5x1xi8>
func.func @negative_extract_strided_slice_2D_to_1D(%arg0 : vector<5x2xi8>) -> vector<5x1xi8> {
  %extracted = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [5, 1], strides = [1, 1]} : vector<5x2xi8> to vector<5x1xi8>
  return %extracted : vector<5x1xi8>
}

// -----

// The 2 elements extracted are contiguous, so this can be expressed as a rank-1 vector.extract_strided_slice.

// CHECK-LABEL: @extract_strided_slice_4D_leading_ones(
//  CHECK-SAME:    %[[A:.*]]: vector<2x1x3x1xi8>) -> vector<1x1x2x1xi8> {
//       CHECK: %[[SC:.*]] = vector.shape_cast %[[A]] : vector<2x1x3x1xi8> to vector<6xi8>
//       CHECK: %[[EXTRACTED:.*]] = vector.extract_strided_slice %[[SC]]
//  CHECK-SAME:    {offsets = [3], sizes = [2], strides = [1]} : vector<6xi8> to vector<2xi8>
//       CHECK: %[[CASTED:.*]] = vector.shape_cast %[[EXTRACTED]] : vector<2xi8> to vector<1x1x2x1xi8>
//       CHECK: return %[[CASTED]] : vector<1x1x2x1xi8>

func.func @extract_strided_slice_4D_leading_ones(%arg0 : vector<2x1x3x1xi8>) -> vector<1x1x2x1xi8> {
  %extracted = vector.extract_strided_slice %arg0 {offsets = [1, 0, 0, 0], sizes = [1, 1, 2, 1], strides = [1, 1, 1, 1]} : vector<2x1x3x1xi8> to vector<1x1x2x1xi8>
  return %extracted : vector<1x1x2x1xi8>
}

// -----

// CHECK-LABEL: @extract_strided_slice_4D_becomes_2D(
//  CHECK-SAME:    %[[A:.*]]: vector<8x7x6x5xi8>) -> vector<2x7x2x5xi8> {
//       CHECK: %[[SC:.*]] = vector.shape_cast %[[A]] : vector<8x7x6x5xi8> to vector<56x30xi8>
//       CHECK: %[[EXTRACTED:.*]] = vector.extract_strided_slice %[[SC]]
//  CHECK-SAME:    {offsets = [14, 5], sizes = [14, 10], strides = [1, 1]} : vector<56x30xi8> to vector<14x10xi8>
//       CHECK: %[[CASTED:.*]] = vector.shape_cast %[[EXTRACTED]] : vector<14x10xi8> to vector<2x7x2x5xi8>
//      CHECK: return %[[CASTED]] : vector<2x7x2x5xi8>
func.func @extract_strided_slice_4D_becomes_2D(%arg0 : vector<8x7x6x5xi8>) -> vector<2x7x2x5xi8> {
  %extracted = vector.extract_strided_slice %arg0 {offsets = [2, 0, 1, 0], sizes = [2, 7, 2, 5], strides = [1, 1, 1, 1]} : vector<8x7x6x5xi8> to vector<2x7x2x5xi8>
  return %extracted : vector<2x7x2x5xi8>
}

// -----

// CHECK-LABEL: @test_extract_strided_slice_4D(
//  CHECK-SAME:    %[[ARG0:.*]]: vector<2x2x2x2xi8>) -> vector<1x2x1x2xi8> {
//   CHECK: %[[SC:.*]] = vector.shape_cast %[[ARG0]] : vector<2x2x2x2xi8> to vector<4x4xi8>
//   CHECK: %[[EXTRACTED:.*]] = vector.extract_strided_slice %[[SC]]
//  CHECK-SAME:    {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xi8> to vector<2x2xi8>
//   CHECK: %[[CASTED:.*]] = vector.shape_cast %[[EXTRACTED]] : vector<2x2xi8> to vector<1x2x1x2xi8>
//   CHECK: return %[[CASTED]] : vector<1x2x1x2xi8>
func.func @test_extract_strided_slice_4D(%arg0 : vector<2x2x2x2xi8>) -> vector<1x2x1x2xi8> {
  %0 = vector.extract_strided_slice %arg0
    {offsets = [1, 0, 1, 0],
       sizes = [1, 2, 1, 2],
     strides = [1, 1, 1, 1]} : vector<2x2x2x2xi8> to vector<1x2x1x2xi8>
  return %0 : vector<1x2x1x2xi8>
}

// -----

// CHECK-LABEL: @extract_strided_slice_4D_becomes_3D(
//  CHECK-SAME:    %[[A:.*]]: vector<8x7x6x5xi8>) -> vector<8x2x6x2xi8> {
//       CHECK: %[[SC:.*]] = vector.shape_cast %[[A]] : vector<8x7x6x5xi8> to vector<8x42x5xi8>
//       CHECK: %[[EXTRACTED:.*]] = vector.extract_strided_slice %[[SC]]
//  CHECK-SAME:    {offsets = [0, 12, 1], sizes = [8, 12, 2], strides = [1, 1, 1]} : vector<8x42x5xi8> to vector<8x12x2xi8>
//       CHECK: %[[CASTED:.*]] = vector.shape_cast %[[EXTRACTED]] : vector<8x12x2xi8> to vector<8x2x6x2xi8>
//       CHECK: return %[[CASTED]] : vector<8x2x6x2xi8>

func.func @extract_strided_slice_4D_becomes_3D(%arg0 : vector<8x7x6x5xi8>) -> vector<8x2x6x2xi8> {
  %extracted = vector.extract_strided_slice %arg0 {offsets = [0, 2, 0, 1], sizes = [8, 2, 6, 2], strides = [1, 1, 1, 1]} : vector<8x7x6x5xi8> to vector<8x2x6x2xi8>
  return %extracted : vector<8x2x6x2xi8>
}

// -----

// CHECK-LABEL: @extract_strided_implicit(
//  CHECK-SAME:    %[[ARG:.*]]: vector<4x8x16xf32>) -> vector<1x8x16xf32> {
//       CHECK: %[[SC0:.*]] = vector.shape_cast %[[ARG]] : vector<4x8x16xf32> to vector<512xf32>
//       CHECK: %[[EXTRACTED:.*]] = vector.extract_strided_slice %[[SC0]]
//  CHECK-SAME:    {offsets = [256], sizes = [128], strides = [1]} : vector<512xf32> to vector<128xf32>
//       CHECK: %[[CASTED:.*]] = vector.shape_cast %[[EXTRACTED]] : vector<128xf32> to vector<1x8x16xf32>
//       CHECK: return %[[CASTED]] : vector<1x8x16xf32>
func.func @extract_strided_implicit(%arg0 : vector<4x8x16xf32>) -> vector<1x8x16xf32> {
  %0 = vector.extract_strided_slice %arg0
      {offsets = [2], sizes = [1], strides = [1]}:
    vector<4x8x16xf32> to vector<1x8x16xf32>
    return %0 : vector<1x8x16xf32>
}

// -----

// **---------------------------------------------**
//       Tests of vector.insert_strided_slice
// **---------------------------------------------**


// CHECK-LABEL: @negative_insert_strided_slice(
//  CHECK-SAME:    %[[A:.*]]: vector<2x2xi8>, %[[B:.*]]: vector<2x1xi8>) -> vector<2x2xi8> {
//       CHECK: %[[INSERTED:.*]] = vector.insert_strided_slice %[[B]], %[[A]]
//       CHECK: return %[[INSERTED]] : vector<2x2xi8>
func.func @negative_insert_strided_slice(%arg0 : vector<2x2xi8>, %arg1 : vector<2x1xi8>) -> vector<2x2xi8> {
  %inserted = vector.insert_strided_slice %arg1, %arg0 {offsets = [0, 1], strides = [1, 1]} : vector<2x1xi8> into vector<2x2xi8>
  return %inserted : vector<2x2xi8>
}

// -----

// CHECK-LABEL: @positive_insert_strided_slice(
//  CHECK-SAME:    %[[A:.*]]: vector<2x2xi8>, %[[B:.*]]: vector<1x2xi8>) -> vector<2x2xi8> {
//   CHECK-DAG: %[[SCA:.*]] = vector.shape_cast %[[A]] : vector<2x2xi8> to vector<4xi8>
//   CHECK-DAG: %[[SCB:.*]] = vector.shape_cast %[[B]] : vector<1x2xi8> to vector<2xi8>
//       CHECK: %[[INSERTED:.*]] = vector.insert_strided_slice %[[SCB]], %[[SCA]]
//  CHECK-SAME:    {offsets = [0], strides = [1]} : vector<2xi8> into vector<4xi8>
//       CHECK: %[[CASTED:.*]] = vector.shape_cast %[[INSERTED]] : vector<4xi8> to vector<2x2xi8>
//       CHECK: return %[[CASTED]] : vector<2x2xi8>

func.func @positive_insert_strided_slice(%arg0 : vector<2x2xi8>, %arg1 : vector<1x2xi8>) -> vector<2x2xi8> {
  %inserted = vector.insert_strided_slice %arg1, %arg0 {offsets = [0, 0], strides = [1, 1]} : vector<1x2xi8> into vector<2x2xi8>
  return %inserted : vector<2x2xi8>
}

// -----

// CHECK-LABEL: @test_insert_strided_slice_4D(
//  CHECK-SAME:    %[[ARG0:.*]]: vector<2x2x2x2xi8>, %[[ARG1:.*]]: vector<1x2x1x2xi8>) -> vector<2x2x2x2xi8> {
//   CHECK-DAG: %[[SC1:.*]] = vector.shape_cast %[[ARG1]] : vector<1x2x1x2xi8> to vector<2x2xi8>
//   CHECK-DAG: %[[SC0:.*]] = vector.shape_cast %[[ARG0]] : vector<2x2x2x2xi8> to vector<4x4xi8>
//       CHECK: %[[INSERTED:.*]] = vector.insert_strided_slice %[[SC1]], %[[SC0]]
//  CHECK-SAME:    {offsets = [2, 2], strides = [1, 1]} : vector<2x2xi8> into vector<4x4xi8>
//       CHECK: %[[CASTED:.*]] = vector.shape_cast %[[INSERTED]] : vector<4x4xi8> to vector<2x2x2x2xi8>
//       CHECK: return %[[CASTED]] : vector<2x2x2x2xi8>
func.func @test_insert_strided_slice_4D(%arg0 : vector<2x2x2x2xi8>, %arg1 : vector<1x2x1x2xi8>) -> vector<2x2x2x2xi8> {
  %0 = vector.insert_strided_slice %arg1, %arg0
    {offsets = [1, 0, 1, 0],
     strides = [1, 1, 1, 1]} : vector<1x2x1x2xi8> into vector<2x2x2x2xi8>
  return %0 : vector<2x2x2x2xi8>
}

// -----

// CHECK-LABEL: @test_insert_strided_implicit_2_into_3(
//  CHECK-SAME:    %[[ARG0:.*]]: vector<16x4x8xf32>, %[[ARG1:.*]]: vector<2x8xf32>) -> vector<16x4x8xf32> {
//   CHECK-DAG: %[[SC1:.*]] = vector.shape_cast %[[ARG1]] : vector<2x8xf32> to vector<16xf32>
//   CHECK-DAG: %[[SC0:.*]] = vector.shape_cast %[[ARG0]] : vector<16x4x8xf32> to vector<512xf32>
//       CHECK: %[[INSERTED:.*]] = vector.insert_strided_slice %[[SC1]], %[[SC0]]
//  CHECK-SAME:    {offsets = [72], strides = [1]} : vector<16xf32> into vector<512xf32>
//       CHECK: %[[CASTED:.*]] = vector.shape_cast %[[INSERTED]] : vector<512xf32> to vector<16x4x8xf32>
//       CHECK: return %[[CASTED]] : vector<16x4x8xf32>

func.func @test_insert_strided_implicit_2_into_3(%arg0 : vector<16x4x8xf32>, %arg1 : vector<2x8xf32>) -> vector<16x4x8xf32> {
   %0 = vector.insert_strided_slice %arg1, %arg0 {offsets = [2, 1, 0], strides = [1, 1]}:
          vector<2x8xf32> into vector<16x4x8xf32>
    return %0 : vector<16x4x8xf32>
}

// -----

// CHECK-LABEL: @test_insert_strided_implicit_1_into_3(
//  CHECK-SAME:    %[[ARG0:.*]]: vector<16x4x8xf32>, %[[ARG1:.*]]: vector<1xf32>) -> vector<16x4x8xf32> {
//       CHECK: %[[SC:.*]] = vector.shape_cast %[[ARG0]] : vector<16x4x8xf32> to vector<512xf32>
//       CHECK: %[[INSERTED:.*]] = vector.insert_strided_slice %[[ARG1]], %[[SC]]
//  CHECK-SAME:    {offsets = [72], strides = [1]} : vector<1xf32> into vector<512xf32>
//       CHECK: %[[CASTED:.*]] = vector.shape_cast %[[INSERTED]] : vector<512xf32> to vector<16x4x8xf32>
//       CHECK: return %[[CASTED]] : vector<16x4x8xf32>

func.func @test_insert_strided_implicit_1_into_3(%arg0 : vector<16x4x8xf32>, %arg1 : vector<1xf32>) -> vector<16x4x8xf32> {
   %0 = vector.insert_strided_slice %arg1, %arg0 {offsets = [2, 1, 0], strides = [1]}:
          vector<1xf32> into vector<16x4x8xf32>
    return %0 : vector<16x4x8xf32>
}
