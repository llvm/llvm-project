// Everything becomes a shuffle (except rank-1 insert/extract).
// RUN: mlir-opt %s -split-input-file -test-vector-linearize=preference=Shuffle | FileCheck %s --check-prefixes=SHUFFLE,ALL

// RUN: mlir-opt %s -split-input-file -test-vector-linearize=preference=Strided | FileCheck %s --check-prefixes=STRIDED,ALL


// **--------------------------------------------------------**
//              Tests of vector.insert
// **--------------------------------------------------------**

// vector.insert where the destination is a 1D vector is always unchanged.
//
// ALL-LABEL: insert_scalar_to_1D(
//  ALL-SAME: %[[A0:.*]]: i8, %[[A1:.*]]: vector<4xi8>
//       ALL: %[[IN0:.*]] = vector.insert %[[A0]], %[[A1]] [2] : i8 into vector<4xi8>
//       ALL: return %[[IN0]] : vector<4xi8>
func.func @insert_scalar_to_1D(%arg0 : i8, %arg1 : vector<4xi8>) -> vector<4xi8>
{
  %inserted = vector.insert %arg0, %arg1[2] : i8 into vector<4xi8>
  return %inserted : vector<4xi8>
}

// -----

// vector.insert of scalar always becomes insert of scalar into 1-D vector.
//
// ALL-LABEL: insert_scalar_to_2D(
//  ALL-SAME: %[[A0:.*]]: i8, %[[A1:.*]]: vector<3x4xi8>
//       ALL: %[[SC0:.*]] = vector.shape_cast %[[A1]] : vector<3x4xi8> to vector<12xi8>
//       ALL: %[[IN0:.*]] = vector.insert %[[A0]], %[[SC0]] [9] : i8 into vector<12xi8>
//       ALL: %[[SC1:.*]] = vector.shape_cast %[[IN0]] : vector<12xi8> to vector<3x4xi8>
//       ALL: return %[[SC1]] : vector<3x4xi8>
func.func @insert_scalar_to_2D(%arg0 : i8, %arg1 : vector<3x4xi8>) -> vector<3x4xi8>
{
  %inserted = vector.insert %arg0, %arg1[2, 1] : i8 into vector<3x4xi8>
  return %inserted : vector<3x4xi8>
}

// -----

// vector.insert where the source isn't a scalar. First case: 1D -> 2D.
//
//    ALL-LABEL: insert_1D_to_2D(
//
//      SHUFFLE: vector.shuffle {{.*}} [0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]
//
//      STRIDED: vector.insert_strided_slice {{.*}} {offsets = [4], strides = [1]}
// STRIDED-SAME: vector<4xi8> into vector<12xi8>
func.func @insert_1D_to_2D(%arg0 : vector<4xi8>, %arg1 : vector<3x4xi8>) -> vector<3x4xi8>
{
  %inserted = vector.insert %arg0, %arg1[1] : vector<4xi8> into vector<3x4xi8>
  return %inserted : vector<3x4xi8>
}


// -----

// vector.insert where the source isn't a scalar. Second case: 0D -> 2D.
//
//    ALL-LABEL: insert_OD_to_2D(
//
//      SHUFFLE: vector.shuffle {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 10, 11] :
// SHUFFLE-SAME: vector<12xi8>, vector<1xi8>
//
//      STRIDED: vector.insert_strided_slice {{.*}} {offsets = [9], strides = [1]}
// STRIDED-SAME: vector<1xi8> into vector<12xi8>
func.func @insert_OD_to_2D(%arg0 : vector<i8>, %arg1 : vector<3x4xi8>) -> vector<3x4xi8>
{
  %inserted = vector.insert %arg0, %arg1[2, 1] : vector<i8> into vector<3x4xi8>
  return %inserted : vector<3x4xi8>
}

// -----

// vector.insert where the source isn't a scalar. Third case: 0D -> 1D.
//
//    ALL-LABEL: insert_OD_to_1D(
//     ALL-SAME: %[[A0:.*]]: vector<i8>, %[[A1:.*]]: vector<4xi8>
//          ALL: %[[IN0:.*]] = vector.insert %[[A0]], %[[A1]] [2] : vector<i8> into vector<4xi8>
//          ALL: return %[[IN0]] : vector<4xi8>
func.func @insert_OD_to_1D(%arg0 : vector<i8>, %arg1 : vector<4xi8>) -> vector<4xi8>
{
  %inserted = vector.insert %arg0, %arg1[2] : vector<i8> into vector<4xi8>
  return %inserted : vector<4xi8>
}

// -----

// vector.insert where the source isn't a scalar. Fourth case: 2D -> 4D.
//
//    ALL-LABEL: insert_2D_to_4D(
//  ALL-COUNT-2: shape_cast
//
//      SHUFFLE: vector.shuffle {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19] :
// SHUFFLE-SAME: vector<16xi8>, vector<4xi8>
//
//      STRIDED: vector.insert_strided_slice {{.*}} {offsets = [12], strides = [1]}
// STRIDED-SAME: vector<4xi8> into vector<16xi8>
func.func @insert_2D_to_4D(%arg0 : vector<2x2xi8>, %arg1 : vector<2x2x2x2xi8>) -> vector<2x2x2x2xi8>
{
  %inserted = vector.insert %arg0, %arg1[1, 1] : vector<2x2xi8> into vector<2x2x2x2xi8>
  return %inserted : vector<2x2x2x2xi8>
}

// -----

// **--------------------------------------------------------**
//              Tests of vector.extract
// **--------------------------------------------------------**

// vector.extract where the source is 1D vector is always unchanged.
//
// ALL-LABEL: extract_scalar_from_1D(
//  ALL-SAME: %[[A0:.*]]: vector<4xi8>
//       ALL: %[[EX0:.*]] = vector.extract %[[A0]][2] : i8 from vector<4xi8>
//       ALL: return %[[EX0]] : i8
func.func @extract_scalar_from_1D(%arg0 : vector<4xi8>) -> i8
{
  %extracted = vector.extract %arg0[2] : i8 from vector<4xi8>
  return %extracted : i8
}

// ALL-LABEL: extract_scalar_from_2D(
//  ALL-SAME: %[[A0:.*]]: vector<3x4xi8>
//       ALL: %[[SC0:.*]] = vector.shape_cast %[[A0]] : vector<3x4xi8> to vector<12xi8>
//       ALL: %[[EX0:.*]] = vector.extract %[[SC0]][9] : i8 from vector<12xi8>
//       ALL: return %[[EX0]] : i8
func.func @extract_scalar_from_2D(%arg0 : vector<3x4xi8>) -> i8
{
  %extracted = vector.extract %arg0[2, 1] : i8 from vector<3x4xi8>
  return %extracted : i8
}

// -----

//    ALL-LABEL: extract_1D_from_2D(
//
//      SHUFFLE: vector.shuffle
// SHUFFLE-SAME: [4, 5, 6, 7] : vector<12xi8>, vector<12xi8>
//
//      STRIDED: vector.extract_strided_slice
// STRIDED-SAME: {offsets = [4], sizes = [4], strides = [1]} : vector<12xi8> to vector<4xi8>
func.func @extract_1D_from_2D(%arg0 : vector<3x4xi8>) -> vector<4xi8>
{
  %extracted = vector.extract %arg0[1] : vector<4xi8> from vector<3x4xi8>
  return %extracted : vector<4xi8>
}

// -----

//    ALL-LABEL: extract_2D_from_4D(
//
//      SHUFFLE: vector.shuffle
// SHUFFLE-SAME: [10, 11] : vector<24xi8>, vector<24xi8>
//
//      STRIDED: vector.extract_strided_slice
// STRIDED-SAME: {offsets = [10], sizes = [2], strides = [1]} : vector<24xi8> to vector<2xi8>
func.func @extract_2D_from_4D(%arg0 : vector<4x3x2x1xi8>) -> vector<2x1xi8> {
  %extracted = vector.extract %arg0[1, 2] : vector<2x1xi8> from vector<4x3x2x1xi8>
  return %extracted : vector<2x1xi8>
}

// **--------------------------------------------------------**
//              Tests of vector.insert_strided_slice
// **--------------------------------------------------------**

// -----

// ALL-LABEL: insert_strided_slice_1D(
//
//   SHUFFLE: shuffle {{.*}} [0, 8, 9, 3, 4, 5, 6, 7]
//
//   STRIDED: insert_strided_slice {{.*}} {offsets = [1], strides = [1]}
func.func @insert_strided_slice_1D(%arg0 : vector<2xi8>, %arg1 : vector<8xi8>) -> vector<8xi8>
{
  %inserted = vector.insert_strided_slice %arg0, %arg1 {offsets = [1], strides = [1]} : vector<2xi8> into vector<8xi8>
  return %inserted : vector<8xi8>
}

// -----

//    ALL-LABEL: insert_strided_slice_4D_contiguous(
//
//      SHUFFLE: vector.shuffle
// SHUFFLE-SAME: 52, 53, 120, 121
// SHUFFLE-SAME: 130, 131, 66, 67
// SHUFFLE-SAME: vector<120xi8>, vector<12xi8>
//
//      STRIDED: vector.insert_strided_slice
// STRIDED-SAME: {offsets = [54], strides = [1]}
// STRIDED-SAME: vector<12xi8> into vector<120xi8>


func.func @insert_strided_slice_4D_contiguous(%arg0 : vector<1x2x2x3xi8>, %arg1 : vector<5x4x2x3xi8>) -> vector<5x4x2x3xi8> {
  %inserted = vector.insert_strided_slice %arg0, %arg1 {offsets = [2, 1, 0, 0], strides = [1, 1, 1, 1]} : vector<1x2x2x3xi8> into vector<5x4x2x3xi8>
  return %inserted : vector<5x4x2x3xi8>
}

// ----- 

// This insert_strided_slice is not contiguous, and so it is always linearized to a 1D vector.shuffle

// ALL-LABEL: insert_strided_slice_4D_noncontiguous(
//       ALL: vector.shuffle
//  ALL-SAME: [0, 1, 2, 8, 4, 5, 6, 9] : vector<8xi8>, vector<2xi8>

func.func @insert_strided_slice_4D_noncontiguous(%arg0 : vector<1x2x1x1xi8>, %arg1 : vector<1x2x2x2xi8>) -> vector<1x2x2x2xi8> {
  %inserted = vector.insert_strided_slice %arg0, %arg1 {offsets = [0, 0, 1, 1], strides = [1, 1, 1, 1]} : vector<1x2x1x1xi8> into vector<1x2x2x2xi8>
  return %inserted : vector<1x2x2x2xi8>
}

// -----

// **--------------------------------------------------------**
//              Tests of vector.extract_strided_slice
// **--------------------------------------------------------**

//    ALL-LABEL: extract_strided_slice_1D(
// 
//      SHUFFLE: vector.shuffle {{.*}} [1, 2] 
// 
//      STRIDED: vector.extract_strided_slice
// STRIDED-SAME: {offsets = [1], sizes = [2], strides = [1]} 
// STRIDED-SAME: vector<8xi8> to vector<2xi8>
func.func @extract_strided_slice_1D(%arg0 : vector<8xi8>) -> vector<2xi8>
{
  %extracted = vector.extract_strided_slice %arg0 {offsets = [1], sizes = [2], strides = [1]} : vector<8xi8> to vector<2xi8>
  return %extracted : vector<2xi8>
}

// ----- 

//    ALL-LABEL: extract_strided_slice_4D_contiguous_1(
//
//      SHUFFLE: vector.shuffle
// SHUFFLE-SAME: [3, 4, 5]
// SHUFFLE-SAME: vector<6xi8>, vector<6xi8>
// 
//     STRIDED: vector.extract_strided_slice
// STRIDED-SAME: {offsets = [3], sizes = [3], strides = [1]}
// STRIDED-SAME: vector<6xi8> to vector<3xi8>
func.func @extract_strided_slice_4D_contiguous_1(%arg0 : vector<2x1x3x1xi8>) -> vector<1x1x3x1xi8> {
  %extracted = vector.extract_strided_slice %arg0 {offsets = [1, 0, 0, 0], sizes = [1, 1, 3, 1], strides = [1, 1, 1, 1]} : vector<2x1x3x1xi8> to vector<1x1x3x1xi8>
  return %extracted : vector<1x1x3x1xi8>
}

// -----  

//    ALL-LABEL: extract_strided_slice_4D_contiguous_2(
// 
//      SHUFFLE: vector.shuffle
// SHUFFLE-SAME: [3, 4]
// SHUFFLE-SAME: vector<6xi8>, vector<6xi8>
//
//      STRIDED: vector.extract_strided_slice
// STRIDED-SAME: {offsets = [3], sizes = [2], strides = [1]}
// STRIDED-SAME: vector<6xi8> to vector<2xi8>
func.func @extract_strided_slice_4D_contiguous_2(%arg0 : vector<2x1x3x1xi8>) -> vector<1x1x2x1xi8> {
  %extracted = vector.extract_strided_slice %arg0 {offsets = [1, 0, 0, 0], sizes = [1, 1, 2, 1], strides = [1, 1, 1, 1]} : vector<2x1x3x1xi8> to vector<1x1x2x1xi8>
  return %extracted : vector<1x1x2x1xi8>
}

// -----

// ALL-LABEL: extract_strided_slice_4D_noncontiguous(
//       ALL: vector.shuffle
//  ALL-SAME: [0, 1, 3, 4]
//  ALL-SAME: vector<6xi8>, vector<6xi8>
func.func @extract_strided_slice_4D_noncontiguous(%arg0 : vector<2x1x3x1xi8>) -> vector<2x1x2x1xi8> {
  %extracted = vector.extract_strided_slice %arg0 {offsets = [0, 0, 0, 0], sizes = [2, 1, 2, 1], strides = [1, 1, 1, 1]} : vector<2x1x3x1xi8> to vector<2x1x2x1xi8>
  return %extracted : vector<2x1x2x1xi8>
}















