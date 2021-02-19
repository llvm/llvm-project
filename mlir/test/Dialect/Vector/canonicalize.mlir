// RUN: mlir-opt %s -pass-pipeline='func(canonicalize)' -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: create_vector_mask_to_constant_mask
func @create_vector_mask_to_constant_mask() -> (vector<4x3xi1>) {
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  // CHECK: vector.constant_mask [3, 2] : vector<4x3xi1>
  %0 = vector.create_mask %c3, %c2 : vector<4x3xi1>
  return %0 : vector<4x3xi1>
}

// -----

func @extract_strided_slice_of_constant_mask() -> (vector<2x2xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x2xi1>
  // CHECK: vector.constant_mask [2, 2] : vector<2x2xi1>
  return %1 : vector<2x2xi1>
}

// -----

func @extract_strided_slice_of_constant_mask() -> (vector<2x2xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [1, 0], sizes = [2, 2], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x2xi1>
  // CHECK: vector.constant_mask [1, 2] : vector<2x2xi1>
  return %1 : vector<2x2xi1>
}

// -----

func @extract_strided_slice_of_constant_mask() -> (vector<2x2xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [0, 1], sizes = [2, 2], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x2xi1>
  // CHECK: vector.constant_mask [2, 1] : vector<2x2xi1>
  return %1 : vector<2x2xi1>
}

// -----

func @extract_strided_slice_of_constant_mask() -> (vector<2x2xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x2xi1>
  // CHECK: vector.constant_mask [0, 0] : vector<2x2xi1>
  return %1 : vector<2x2xi1>
}

// -----

func @extract_strided_slice_of_constant_mask() -> (vector<2x1xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [0, 2], sizes = [2, 1], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x1xi1>
  // CHECK: vector.constant_mask [0, 0] : vector<2x1xi1>
  return %1 : vector<2x1xi1>
}

// -----

func @extract_strided_slice_of_constant_mask() -> (vector<2x1xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [0, 1], sizes = [2, 1], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x1xi1>
  // CHECK: vector.constant_mask [2, 1] : vector<2x1xi1>
  return %1 : vector<2x1xi1>
}

// -----

func @extract_strided_slice_of_constant_mask() -> (vector<2x1xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [1, 1], sizes = [2, 1], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x1xi1>
  // CHECK: vector.constant_mask [1, 1] : vector<2x1xi1>
  return %1 : vector<2x1xi1>
}

// -----

// CHECK-LABEL: extract_strided_fold
//  CHECK-SAME: (%[[ARG:.*]]: vector<4x3xi1>)
//  CHECK-NEXT:   return %[[ARG]] : vector<4x3xi1>
func @extract_strided_fold(%arg : vector<4x3xi1>) -> (vector<4x3xi1>) {
  %0 = vector.extract_strided_slice %arg
    {offsets = [0, 0], sizes = [4, 3], strides = [1, 1]}
      : vector<4x3xi1> to vector<4x3xi1>
  return %0 : vector<4x3xi1>
}

// -----

// CHECK-LABEL: extract_strided_fold_insert
//  CHECK-SAME: (%[[ARG:.*]]: vector<4x4xf32>
//  CHECK-NEXT:   return %[[ARG]] : vector<4x4xf32>
func @extract_strided_fold_insert(%a: vector<4x4xf32>, %b: vector<8x16xf32>)
  -> (vector<4x4xf32>) {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1, 1]}
    : vector<4x4xf32> into vector<8x16xf32>
  %1 = vector.extract_strided_slice %0
    {offsets = [2, 2], sizes = [4, 4], strides = [1, 1]}
      : vector<8x16xf32> to vector<4x4xf32>
  return %1 : vector<4x4xf32>
}

// -----

// Case where the vector inserted is a subset of the vector extracted.
// CHECK-LABEL: extract_strided_fold_insert
//  CHECK-SAME: (%[[ARG0:.*]]: vector<6x4xf32>
//  CHECK-NEXT:   %[[EXT:.*]] = vector.extract_strided_slice %[[ARG0]]
//  CHECK-SAME:     {offsets = [0, 0], sizes = [4, 4], strides = [1, 1]}
//  CHECK-SAME:       : vector<6x4xf32> to vector<4x4xf32>
//  CHECK-NEXT:   return %[[EXT]] : vector<4x4xf32>
func @extract_strided_fold_insert(%a: vector<6x4xf32>, %b: vector<8x16xf32>)
  -> (vector<4x4xf32>) {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1, 1]}
    : vector<6x4xf32> into vector<8x16xf32>
  %1 = vector.extract_strided_slice %0
    {offsets = [2, 2], sizes = [4, 4], strides = [1, 1]}
      : vector<8x16xf32> to vector<4x4xf32>
  return %1 : vector<4x4xf32>
}

// -----

// Negative test where the extract is not a subset of the element inserted.
// CHECK-LABEL: extract_strided_fold_negative
//  CHECK-SAME: (%[[ARG0:.*]]: vector<4x4xf32>, %[[ARG1:.*]]: vector<8x16xf32>
//       CHECK:   %[[INS:.*]] = vector.insert_strided_slice %[[ARG0]], %[[ARG1]]
//  CHECK-SAME:     {offsets = [2, 2], strides = [1, 1]}
//  CHECK-SAME:       : vector<4x4xf32> into vector<8x16xf32>
//       CHECK:   %[[EXT:.*]] = vector.extract_strided_slice %[[INS]]
//  CHECK-SAME:     {offsets = [2, 2], sizes = [6, 4], strides = [1, 1]}
//  CHECK-SAME:       : vector<8x16xf32> to vector<6x4xf32>
//  CHECK-NEXT:   return %[[EXT]] : vector<6x4xf32>
func @extract_strided_fold_negative(%a: vector<4x4xf32>, %b: vector<8x16xf32>)
  -> (vector<6x4xf32>) {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1, 1]}
    : vector<4x4xf32> into vector<8x16xf32>
  %1 = vector.extract_strided_slice %0
    {offsets = [2, 2], sizes = [6, 4], strides = [1, 1]}
      : vector<8x16xf32> to vector<6x4xf32>
  return %1 : vector<6x4xf32>
}

// -----

// Case where we need to go through 2 level of insert element.
// CHECK-LABEL: extract_strided_fold_insert
//  CHECK-SAME: (%[[ARG0:.*]]: vector<2x8xf32>, %[[ARG1:.*]]: vector<1x4xf32>,
//  CHECK-NEXT:   %[[EXT:.*]] = vector.extract_strided_slice %[[ARG1]]
//  CHECK-SAME:     {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]}
//  CHECK-SAME:       : vector<1x4xf32> to vector<1x1xf32>
//  CHECK-NEXT:   return %[[EXT]] : vector<1x1xf32>
func @extract_strided_fold_insert(%a: vector<2x8xf32>, %b: vector<1x4xf32>,
                                  %c : vector<1x4xf32>) -> (vector<1x1xf32>) {
  %0 = vector.insert_strided_slice %b, %a {offsets = [0, 1], strides = [1, 1]}
    : vector<1x4xf32> into vector<2x8xf32>
  %1 = vector.insert_strided_slice %c, %0 {offsets = [1, 0], strides = [1, 1]}
    : vector<1x4xf32> into vector<2x8xf32>
  %2 = vector.extract_strided_slice %1
      {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]}
        : vector<2x8xf32> to vector<1x1xf32>
  return %2 : vector<1x1xf32>
}

// -----

// CHECK-LABEL: transpose_1D_identity
// CHECK-SAME: ([[ARG:%.*]]: vector<4xf32>)
func @transpose_1D_identity(%arg : vector<4xf32>) -> vector<4xf32> {
  // CHECK-NOT: transpose
  %0 = vector.transpose %arg, [0] : vector<4xf32> to vector<4xf32>
  // CHECK-NEXT: return [[ARG]]
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: transpose_2D_identity
// CHECK-SAME: ([[ARG:%.*]]: vector<4x3xf32>)
func @transpose_2D_identity(%arg : vector<4x3xf32>) -> vector<4x3xf32> {
  // CHECK-NOT: transpose
  %0 = vector.transpose %arg, [0, 1] : vector<4x3xf32> to vector<4x3xf32>
  // CHECK-NEXT: return [[ARG]]
  return %0 : vector<4x3xf32>
}

// -----

// CHECK-LABEL: transpose_3D_identity
// CHECK-SAME: ([[ARG:%.*]]: vector<4x3x2xf32>)
func @transpose_3D_identity(%arg : vector<4x3x2xf32>) -> vector<4x3x2xf32> {
  // CHECK-NOT: transpose
  %0 = vector.transpose %arg, [0, 1, 2] : vector<4x3x2xf32> to vector<4x3x2xf32>
  // CHECK-NEXT: return [[ARG]]
  return %0 : vector<4x3x2xf32>
}

// -----

// CHECK-LABEL: transpose_2D_sequence
// CHECK-SAME: ([[ARG:%.*]]: vector<4x3xf32>)
func @transpose_2D_sequence(%arg : vector<4x3xf32>) -> vector<4x3xf32> {
  // CHECK-NOT: transpose
  %0 = vector.transpose %arg, [1, 0] : vector<4x3xf32> to vector<3x4xf32>
  %1 = vector.transpose %0, [0, 1] : vector<3x4xf32> to vector<3x4xf32>
  %2 = vector.transpose %1, [1, 0] : vector<3x4xf32> to vector<4x3xf32>
  %3 = vector.transpose %2, [0, 1] : vector<4x3xf32> to vector<4x3xf32>
  // CHECK: [[ADD:%.*]] = addf [[ARG]], [[ARG]]
  %4 = addf %2, %3 : vector<4x3xf32>
  // CHECK-NEXT: return [[ADD]]
  return %4 : vector<4x3xf32>
}

// -----

// CHECK-LABEL: transpose_3D_sequence
// CHECK-SAME: ([[ARG:%.*]]: vector<4x3x2xf32>)
func @transpose_3D_sequence(%arg : vector<4x3x2xf32>) -> vector<4x3x2xf32> {
  // CHECK: [[T0:%.*]] = vector.transpose [[ARG]], [2, 1, 0]
  %0 = vector.transpose %arg, [1, 2, 0] : vector<4x3x2xf32> to vector<3x2x4xf32>
  %1 = vector.transpose %0, [1, 0, 2] : vector<3x2x4xf32> to vector<2x3x4xf32>
  // CHECK-NOT: transpose
  %2 = vector.transpose %1, [2, 1, 0] : vector<2x3x4xf32> to vector<4x3x2xf32>
  %3 = vector.transpose %2, [2, 1, 0] : vector<4x3x2xf32> to vector<2x3x4xf32>
  // CHECK: [[MUL:%.*]] = mulf [[T0]], [[T0]]
  %4 = mulf %1, %3 : vector<2x3x4xf32>
  // CHECK: [[T5:%.*]] = vector.transpose [[MUL]], [2, 1, 0]
  %5 = vector.transpose %4, [2, 1, 0] : vector<2x3x4xf32> to vector<4x3x2xf32>
  // CHECK-NOT: transpose
  %6 = vector.transpose %3, [2, 1, 0] : vector<2x3x4xf32> to vector<4x3x2xf32>
  // CHECK: [[ADD:%.*]] = addf [[T5]], [[ARG]]
  %7 = addf %5, %6 : vector<4x3x2xf32>
  // CHECK-NEXT: return [[ADD]]
  return %7 : vector<4x3x2xf32>
}

// -----

// CHECK-LABEL: cast_transfers
func @cast_transfers(%A: memref<4x8xf32>) -> (vector<4x8xf32>) {
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32
  %0 = memref_cast %A : memref<4x8xf32> to memref<?x?xf32>

  // CHECK: vector.transfer_read %{{.*}} {masked = [false, false]} : memref<4x8xf32>, vector<4x8xf32>
  %1 = vector.transfer_read %0[%c0, %c0], %f0 : memref<?x?xf32>, vector<4x8xf32>

  // CHECK: vector.transfer_write %{{.*}} {masked = [false, false]} : vector<4x8xf32>, memref<4x8xf32>
  vector.transfer_write %1, %0[%c0, %c0] : vector<4x8xf32>, memref<?x?xf32>
  return %1 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: cast_transfers
func @cast_transfers(%A: tensor<4x8xf32>) -> (vector<4x8xf32>) {
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32
  %0 = tensor.cast %A : tensor<4x8xf32> to tensor<?x?xf32>

  // CHECK: vector.transfer_read %{{.*}} {masked = [false, false]} : tensor<4x8xf32>, vector<4x8xf32>
  %1 = vector.transfer_read %0[%c0, %c0], %f0 : tensor<?x?xf32>, vector<4x8xf32>

  return %1 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func @insert_extract_transpose_2d(
//  CHECK-SAME: %[[V:[a-zA-Z0-9]*]]: vector<2x3xf32>,
//  CHECK-SAME: %[[F0:[a-zA-Z0-9]*]]: f32,
//  CHECK-SAME: %[[F1:[a-zA-Z0-9]*]]: f32,
//  CHECK-SAME: %[[F2:[a-zA-Z0-9]*]]: f32,
//  CHECK-SAME: %[[F3:[a-zA-Z0-9]*]]: f32
func @insert_extract_transpose_2d(
    %v: vector<2x3xf32>, %f0: f32, %f1: f32, %f2: f32, %f3: f32)
-> (f32, f32, f32)
{
  %0 = vector.insert %f0, %v[0, 0] : f32 into vector<2x3xf32>
  %1 = vector.insert %f1, %0[0, 1] : f32 into vector<2x3xf32>
  %2 = vector.insert %f2, %1[1, 0] : f32 into vector<2x3xf32>
  %3 = vector.insert %f3, %2[1, 1] : f32 into vector<2x3xf32>
  %4 = vector.transpose %3, [1, 0] : vector<2x3xf32> to vector<3x2xf32>
  %5 = vector.insert %f3, %4[1, 0] : f32 into vector<3x2xf32>
  %6 = vector.transpose %5, [1, 0] : vector<3x2xf32> to vector<2x3xf32>

  // Expected %f2 from %2 = vector.insert %f2, %1[1, 0].
  %r1 = vector.extract %3[1, 0] : vector<2x3xf32>

  // Expected %f1 from %1 = vector.insert %f1, %0[0, 1] followed by
  // transpose [1, 0].
  %r2 = vector.extract %4[1, 0] : vector<3x2xf32>

  // Expected %f2 from %2 = vector.insert %f2, %1[1, 0] followed by double
  // transpose [1, 0].
  %r3 = vector.extract %6[1, 0] : vector<2x3xf32>

  // CHECK-NEXT: return %[[F2]], %[[F1]], %[[F2]] : f32, f32, f32
  return %r1, %r2, %r3 : f32, f32, f32
}

// -----

// CHECK-LABEL: func @insert_extract_transpose_3d(
//  CHECK-SAME: %[[V:[a-zA-Z0-9]*]]: vector<2x3x4xf32>,
//  CHECK-SAME: %[[F0:[a-zA-Z0-9]*]]: f32,
//  CHECK-SAME: %[[F1:[a-zA-Z0-9]*]]: f32,
//  CHECK-SAME: %[[F2:[a-zA-Z0-9]*]]: f32,
//  CHECK-SAME: %[[F3:[a-zA-Z0-9]*]]: f32
func @insert_extract_transpose_3d(
    %v: vector<2x3x4xf32>, %f0: f32, %f1: f32, %f2: f32, %f3: f32)
-> (f32, f32, f32, f32)
{
  %0 = vector.insert %f0, %v[0, 0, 0] : f32 into vector<2x3x4xf32>
  %1 = vector.insert %f1, %0[0, 1, 0] : f32 into vector<2x3x4xf32>
  %2 = vector.insert %f2, %1[1, 0, 0] : f32 into vector<2x3x4xf32>
  %3 = vector.insert %f3, %2[0, 0, 1] : f32 into vector<2x3x4xf32>
  %4 = vector.transpose %3, [1, 2, 0] : vector<2x3x4xf32> to vector<3x4x2xf32>
  %5 = vector.insert %f3, %4[1, 0, 0] : f32 into vector<3x4x2xf32>
  %6 = vector.transpose %5, [1, 2, 0] : vector<3x4x2xf32> to vector<4x2x3xf32>
  %7 = vector.insert %f3, %6[1, 0, 0] : f32 into vector<4x2x3xf32>
  %8 = vector.transpose %7, [1, 2, 0] : vector<4x2x3xf32> to vector<2x3x4xf32>

  // Expected %f2 from %2 = vector.insert %f2, %1[1, 0, 0].
  %r1 = vector.extract %3[1, 0, 0] : vector<2x3x4xf32>

  // Expected %f1 from %1 = vector.insert %f1, %0[0, 1, 0] followed by
  // transpose[1, 2, 0].
  %r2 = vector.extract %4[1, 0, 0] : vector<3x4x2xf32>

  // Expected %f3 from %3 = vector.insert %f3, %0[0, 0, 1] followed by double
  // transpose[1, 2, 0].
  %r3 = vector.extract %6[1, 0, 0] : vector<4x2x3xf32>

  // Expected %f2 from %2 = vector.insert %f2, %1[1, 0, 0] followed by triple
  // transpose[1, 2, 0].
  %r4 = vector.extract %8[1, 0, 0] : vector<2x3x4xf32>

  // CHECK-NEXT: return %[[F2]], %[[F1]], %[[F3]], %[[F2]] : f32, f32, f32
  return %r1, %r2, %r3, %r4 : f32, f32, f32, f32
}

// -----

// CHECK-LABEL: func @insert_extract_transpose_3d_2d(
//  CHECK-SAME: %[[V:[a-zA-Z0-9]*]]: vector<2x3x4xf32>,
//  CHECK-SAME: %[[F0:[a-zA-Z0-9]*]]: vector<4xf32>,
//  CHECK-SAME: %[[F1:[a-zA-Z0-9]*]]: vector<4xf32>,
//  CHECK-SAME: %[[F2:[a-zA-Z0-9]*]]: vector<4xf32>,
//  CHECK-SAME: %[[F3:[a-zA-Z0-9]*]]: vector<4xf32>
func @insert_extract_transpose_3d_2d(
    %v: vector<2x3x4xf32>,
    %f0: vector<4xf32>, %f1: vector<4xf32>, %f2: vector<4xf32>, %f3: vector<4xf32>)
-> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
{
  %0 = vector.insert %f0, %v[0, 0] : vector<4xf32> into vector<2x3x4xf32>
  %1 = vector.insert %f1, %0[0, 1] : vector<4xf32> into vector<2x3x4xf32>
  %2 = vector.insert %f2, %1[1, 0] : vector<4xf32> into vector<2x3x4xf32>
  %3 = vector.insert %f3, %2[1, 1] : vector<4xf32> into vector<2x3x4xf32>
  %4 = vector.transpose %3, [1, 0, 2] : vector<2x3x4xf32> to vector<3x2x4xf32>
  %5 = vector.transpose %4, [1, 0, 2] : vector<3x2x4xf32> to vector<2x3x4xf32>

  // Expected %f2 from %2 = vector.insert %f2, %1[1, 0].
  %r1 = vector.extract %3[1, 0] : vector<2x3x4xf32>

  // Expected %f1 from %1 = vector.insert %f1, %0[0, 1] followed by
  // transpose[1, 0, 2].
  %r2 = vector.extract %4[1, 0] : vector<3x2x4xf32>

  // Expected %f2 from %2 = vector.insert %f2, %1[1, 0, 0] followed by double
  // transpose[1, 0, 2].
  %r3 = vector.extract %5[1, 0] : vector<2x3x4xf32>

  %6 = vector.transpose %3, [1, 2, 0] : vector<2x3x4xf32> to vector<3x4x2xf32>
  %7 = vector.transpose %6, [1, 2, 0] : vector<3x4x2xf32> to vector<4x2x3xf32>
  %8 = vector.transpose %7, [1, 2, 0] : vector<4x2x3xf32> to vector<2x3x4xf32>

  // Expected %f2 from %2 = vector.insert %f2, %1[1, 0, 0] followed by triple
  // transpose[1, 2, 0].
  %r4 = vector.extract %8[1, 0] : vector<2x3x4xf32>

  //      CHECK: return %[[F2]], %[[F1]], %[[F2]], %[[F2]]
  // CHECK-SAME: vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
  return %r1, %r2, %r3, %r4 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
}

// -----

// CHECK-LABEL: fold_extracts
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: vector<3x4x5x6xf32>
func @fold_extracts(%a : vector<3x4x5x6xf32>) -> (f32, vector<4x5x6xf32>) {
  %b = vector.extract %a[0] : vector<3x4x5x6xf32>
  %c = vector.extract %b[1, 2] : vector<4x5x6xf32>
  //  CHECK-NEXT: vector.extract %[[A]][0, 1, 2, 3] : vector<3x4x5x6xf32>
  %d = vector.extract %c[3] : vector<6xf32>

  //  CHECK-NEXT: vector.extract %[[A]][0] : vector<3x4x5x6xf32>
  %e = vector.extract %a[0] : vector<3x4x5x6xf32>

  //  CHECK-NEXT: return
  return %d, %e : f32, vector<4x5x6xf32>
}

// -----

// CHECK-LABEL: fold_extract_transpose
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: vector<3x4x5x6xf32>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]*]]: vector<3x6x5x6xf32>
func @fold_extract_transpose(
    %a : vector<3x4x5x6xf32>, %b : vector<3x6x5x6xf32>) -> (
      vector<6xf32>, vector<6xf32>, vector<6xf32>) {
  // [3] is a proper most minor identity map in transpose.
  // Permutation is a self inverse and we have.
  // [0, 2, 1] ^ -1 o [0, 1, 2] = [0, 2, 1] o [0, 1, 2]
  //                            = [0, 2, 1]
  //  CHECK-NEXT: vector.extract %[[A]][0, 2, 1] : vector<3x4x5x6xf32>
  %0 = vector.transpose %a, [0, 2, 1, 3] : vector<3x4x5x6xf32> to vector<3x5x4x6xf32>
  %1 = vector.extract %0[0, 1, 2] : vector<3x5x4x6xf32>

  // [3] is a proper most minor identity map in transpose.
  // Permutation is a not self inverse and we have.
  // [1, 2, 0] ^ -1 o [0, 1, 2] = [2, 0, 1] o [0, 1, 2]
  //                            = [2, 0, 1]
  //  CHECK-NEXT: vector.extract %[[A]][2, 0, 1] : vector<3x4x5x6xf32>
  %2 = vector.transpose %a, [1, 2, 0, 3] : vector<3x4x5x6xf32> to vector<4x5x3x6xf32>
  %3 = vector.extract %2[0, 1, 2] : vector<4x5x3x6xf32>

  // Not a minor identity map so intra-vector level has been permuted
  //  CHECK-NEXT: vector.transpose %[[B]], [0, 2, 3, 1]
  //  CHECK-NEXT: vector.extract %{{.*}}[0, 1, 2]
  %4 = vector.transpose %b, [0, 2, 3, 1] : vector<3x6x5x6xf32> to vector<3x5x6x6xf32>
  %5 = vector.extract %4[0, 1, 2] : vector<3x5x6x6xf32>

  return %1, %3, %5 : vector<6xf32>, vector<6xf32>, vector<6xf32>
}

// -----

// CHECK-LABEL: fold_extract_broadcast
//  CHECK-SAME:   %[[A:.*]]: f32
//       CHECK:   return %[[A]] : f32
func @fold_extract_broadcast(%a : f32) -> f32 {
  %b = vector.broadcast %a : f32 to vector<1x2x4xf32>
  %r = vector.extract %b[0, 1, 2] : vector<1x2x4xf32>
  return %r : f32
}

// -----

// CHECK-LABEL: fold_extract_broadcast_vector
//  CHECK-SAME:   %[[A:.*]]: vector<4xf32>
//       CHECK:   return %[[A]] : vector<4xf32>
func @fold_extract_broadcast_vector(%a : vector<4xf32>) -> vector<4xf32> {
  %b = vector.broadcast %a : vector<4xf32> to vector<1x2x4xf32>
  %r = vector.extract %b[0, 1] : vector<1x2x4xf32>
  return %r : vector<4xf32>
}

// -----

// CHECK-LABEL: fold_extract_broadcast
//  CHECK-SAME:   %[[A:.*]]: vector<4xf32>
//       CHECK:   %[[R:.*]] = vector.extract %[[A]][2] : vector<4xf32>
//       CHECK:   return %[[R]] : f32
func @fold_extract_broadcast(%a : vector<4xf32>) -> f32 {
  %b = vector.broadcast %a : vector<4xf32> to vector<1x2x4xf32>
  %r = vector.extract %b[0, 1, 2] : vector<1x2x4xf32>
  return %r : f32
}

// -----

// Negative test for extract_op folding when the type of broadcast source
// doesn't match the type of vector.extract.
// CHECK-LABEL: fold_extract_broadcast_negative
//       CHECK:   %[[B:.*]] = vector.broadcast %{{.*}} : f32 to vector<1x2x4xf32>
//       CHECK:   %[[R:.*]] = vector.extract %[[B]][0, 1] : vector<1x2x4xf32>
//       CHECK:   return %[[R]] : vector<4xf32>
func @fold_extract_broadcast_negative(%a : f32) -> vector<4xf32> {
  %b = vector.broadcast %a : f32 to vector<1x2x4xf32>
  %r = vector.extract %b[0, 1] : vector<1x2x4xf32>
  return %r : vector<4xf32>
}

// -----

// CHECK-LABEL: func @fold_extract_shapecast
//  CHECK-SAME: (%[[A0:.*]]: vector<5x1x3x2xf32>, %[[A1:.*]]: vector<8x4x2xf32>
//       CHECK:   %[[R0:.*]] = vector.extract %[[A0]][1, 0, 1, 1] : vector<5x1x3x2xf32>
//       CHECK:   %[[R1:.*]] = vector.extract %[[A0]][1, 0, 2] : vector<5x1x3x2xf32>
//       CHECK:   %[[R2:.*]] = vector.extract %[[A1]][7] : vector<8x4x2xf32>
//       CHECK:   return %[[R0]], %[[R1]], %[[R2]] : f32, vector<2xf32>, vector<4x2xf32>
func @fold_extract_shapecast(%arg0 : vector<5x1x3x2xf32>,
                             %arg1 : vector<8x4x2xf32>)
  -> (f32, vector<2xf32>, vector<4x2xf32>) {
  %0 = vector.shape_cast %arg0 : vector<5x1x3x2xf32> to vector<15x2xf32>
  %1 = vector.shape_cast %arg1 : vector<8x4x2xf32> to vector<4x2x4x2xf32>
  %r1 = vector.extract %0[4, 1] : vector<15x2xf32>
  %r2 = vector.extract %0[5] : vector<15x2xf32>
  %r3 = vector.extract %1[3, 1] : vector<4x2x4x2xf32>
  return %r1, %r2, %r3 : f32, vector<2xf32>, vector<4x2xf32>
}

// -----

// CHECK-LABEL: fold_extract_shapecast_negative
//       CHECK:   %[[V:.*]] = vector.shape_cast %{{.*}} : vector<16xf32> to vector<2x4x2xf32>
//       CHECK:   %[[R:.*]] = vector.extract %[[V]][1] : vector<2x4x2xf32>
//       CHECK:   return %[[R]] : vector<4x2xf32>
func @fold_extract_shapecast_negative(%arg0 : vector<16xf32>,
                             %arg1 : vector<8x4x2xf32>) -> vector<4x2xf32> {
  %0 = vector.shape_cast %arg0 : vector<16xf32> to vector<2x4x2xf32>
  %r = vector.extract %0[1] : vector<2x4x2xf32>
  return %r : vector<4x2xf32>
}


// -----

// CHECK-LABEL: fold_vector_transfers
func @fold_vector_transfers(%A: memref<?x8xf32>) -> (vector<4x8xf32>, vector<4x9xf32>) {
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32

  // CHECK: vector.transfer_read %{{.*}} {masked = [true, false]}
  %1 = vector.transfer_read %A[%c0, %c0], %f0 : memref<?x8xf32>, vector<4x8xf32>

  // CHECK: vector.transfer_write %{{.*}} {masked = [true, false]}
  vector.transfer_write %1, %A[%c0, %c0] : vector<4x8xf32>, memref<?x8xf32>

  // Both dims masked, attribute is elided.
  // CHECK: vector.transfer_read %{{.*}}
  // CHECK-NOT: masked
  %2 = vector.transfer_read %A[%c0, %c0], %f0 : memref<?x8xf32>, vector<4x9xf32>

  // Both dims masked, attribute is elided.
  // CHECK: vector.transfer_write %{{.*}}
  // CHECK-NOT: masked
  vector.transfer_write %2, %A[%c0, %c0] : vector<4x9xf32>, memref<?x8xf32>

  // CHECK: return
  return %1, %2 : vector<4x8xf32>, vector<4x9xf32>
}

// -----

// CHECK-LABEL: bitcast_folding
//  CHECK-SAME:   %[[A:.*]]: vector<4x8xf32>
//  CHECK-SAME:   %[[B:.*]]: vector<2xi32>
//  CHECK:        return %[[A]], %[[B]] : vector<4x8xf32>, vector<2xi32>
func @bitcast_folding(%I1: vector<4x8xf32>, %I2: vector<2xi32>) -> (vector<4x8xf32>, vector<2xi32>) {
  %0 = vector.bitcast %I1 : vector<4x8xf32> to vector<4x8xf32>
  %1 = vector.bitcast %I2 : vector<2xi32> to vector<4xi16>
  %2 = vector.bitcast %1 : vector<4xi16> to vector<2xi32>
  return %0, %2 : vector<4x8xf32>, vector<2xi32>
}

// CHECK-LABEL: func @bitcast_f16_to_f32
//              bit pattern: 0x00000000
//       CHECK: %[[CST0:.+]] = constant dense<0.000000e+00> : vector<4xf32>
//              bit pattern: 0x40004000
//       CHECK: %[[CST1:.+]] = constant dense<2.00390625> : vector<4xf32>
//       CHECK: return %[[CST0]], %[[CST1]]
func @bitcast_f16_to_f32() -> (vector<4xf32>, vector<4xf32>) {
  %cst0 = constant dense<0.0> : vector<8xf16> // bit pattern: 0x0000
  %cst1 = constant dense<2.0> : vector<8xf16> // bit pattern: 0x4000
  %cast0 = vector.bitcast %cst0: vector<8xf16> to vector<4xf32>
  %cast1 = vector.bitcast %cst1: vector<8xf16> to vector<4xf32>
  return %cast0, %cast1: vector<4xf32>, vector<4xf32>
}

// -----

// CHECK-LABEL: broadcast_folding1
//       CHECK: %[[CST:.*]] = constant dense<42> : vector<4xi32>
//   CHECK-NOT: vector.broadcast
//       CHECK: return %[[CST]]
func @broadcast_folding1() -> vector<4xi32> {
  %0 = constant 42 : i32
  %1 = vector.broadcast %0 : i32 to vector<4xi32>
  return %1 : vector<4xi32>
}

// -----

// CHECK-LABEL: @broadcast_folding2
//       CHECK: %[[CST:.*]] = constant dense<42> : vector<4x16xi32>
//   CHECK-NOT: vector.broadcast
//       CHECK: return %[[CST]]
func @broadcast_folding2() -> vector<4x16xi32> {
  %0 = constant 42 : i32
  %1 = vector.broadcast %0 : i32 to vector<16xi32>
  %2 = vector.broadcast %1 : vector<16xi32> to vector<4x16xi32>
  return %2 : vector<4x16xi32>
}

// -----

// CHECK-LABEL: shape_cast_constant
//       CHECK: %[[CST0:.*]] = constant dense<2.000000e+00> : vector<20x2xf32>
//       CHECK: %[[CST1:.*]] = constant dense<1> : vector<3x4x2xi32>
//       CHECK: return %[[CST0]], %[[CST1]] : vector<20x2xf32>, vector<3x4x2xi32>
func @shape_cast_constant() -> (vector<20x2xf32>, vector<3x4x2xi32>) {
  %cst = constant dense<2.000000e+00> : vector<5x4x2xf32>
  %cst_1 = constant dense<1> : vector<12x2xi32>
  %0 = vector.shape_cast %cst : vector<5x4x2xf32> to vector<20x2xf32>
  %1 = vector.shape_cast %cst_1 : vector<12x2xi32> to vector<3x4x2xi32>
  return %0, %1 : vector<20x2xf32>, vector<3x4x2xi32>
}

// -----

// CHECK-LABEL: extract_strided_constant
//       CHECK: %[[CST0:.*]] = constant dense<2.000000e+00> : vector<12x2xf32>
//       CHECK: %[[CST1:.*]] = constant dense<1> : vector<2x13x3xi32>
//       CHECK: return %[[CST0]], %[[CST1]] : vector<12x2xf32>, vector<2x13x3xi32>
func @extract_strided_constant() -> (vector<12x2xf32>, vector<2x13x3xi32>) {
  %cst = constant dense<2.000000e+00> : vector<29x7xf32>
  %cst_1 = constant dense<1> : vector<4x37x9xi32>
  %0 = vector.extract_strided_slice %cst
    {offsets = [2, 3], sizes = [12, 2], strides = [1, 1]}
      : vector<29x7xf32> to vector<12x2xf32>
  %1 = vector.extract_strided_slice %cst_1
    {offsets = [1, 2, 5], sizes = [2, 13, 3], strides = [1, 1, 1]}
      : vector<4x37x9xi32> to vector<2x13x3xi32>
  return %0, %1 : vector<12x2xf32>, vector<2x13x3xi32>
}

// -----

// CHECK-LABEL: extract_strided_broadcast
//       CHECK:   %[[B:.*]] = vector.broadcast %{{.*}} : vector<4xf16> to vector<2x4xf16>
//  CHECK-NEXT:   return %[[B]] : vector<2x4xf16>
func @extract_strided_broadcast(%arg0: vector<4xf16>) -> vector<2x4xf16> {
 %0 = vector.broadcast %arg0 : vector<4xf16> to vector<16x4xf16>
 %1 = vector.extract_strided_slice %0
  {offsets = [0, 0], sizes = [2, 4], strides = [1, 1]} :
  vector<16x4xf16> to vector<2x4xf16>
  return %1 : vector<2x4xf16>
}

// -----

// CHECK-LABEL: extract_strided_broadcast2
//       CHECK:   %[[E:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0], sizes = [2], strides = [1]} : vector<4xf16> to vector<2xf16>
//  CHECK-NEXT:   %[[B:.*]] = vector.broadcast %[[E]] : vector<2xf16> to vector<2x2xf16>
//  CHECK-NEXT:   return %[[B]] : vector<2x2xf16>
func @extract_strided_broadcast2(%arg0: vector<4xf16>) -> vector<2x2xf16> {
 %0 = vector.broadcast %arg0 : vector<4xf16> to vector<16x4xf16>
 %1 = vector.extract_strided_slice %0
  {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} :
  vector<16x4xf16> to vector<2x2xf16>
  return %1 : vector<2x2xf16>
}

// -----

// CHECK-LABEL: consecutive_shape_cast
//       CHECK:   %[[C:.*]] = vector.shape_cast %{{.*}} : vector<16xf16> to vector<4x4xf16>
//  CHECK-NEXT:   return %[[C]] : vector<4x4xf16>
func @consecutive_shape_cast(%arg0: vector<16xf16>) -> vector<4x4xf16> {
  %0 = vector.shape_cast %arg0 : vector<16xf16> to vector<2x8xf16>
  %1 = vector.shape_cast %0 : vector<2x8xf16> to vector<4x4xf16>
  return %1 : vector<4x4xf16>
}

// -----

// CHECK-LABEL: broadcast_to_shapecast
//       CHECK:   %[[C:.*]] = vector.shape_cast %{{.*}} : vector<4x4xf16> to vector<1x4x4xf16>
//  CHECK-NEXT:   return %[[C]] : vector<1x4x4xf16>
func @broadcast_to_shapecast(%arg0: vector<4x4xf16>) -> vector<1x4x4xf16> {
  %0 = vector.broadcast %arg0 : vector<4x4xf16> to vector<1x4x4xf16>
  return %0 : vector<1x4x4xf16>
}

// -----

// CHECK-LABEL: func @dead_transfer_op
//   CHECK-NOT:   vector.transfer_read
//   CHECK-NOT:   vector.transfer_write
//       CHECK:   return
func @dead_transfer_op(%arg0 : tensor<4x4xf32>, %arg1 : memref<4x4xf32>,
                       %v0 : vector<1x4xf32>) {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %r = vector.transfer_read %arg1[%c0, %c0], %cf0 :
    memref<4x4xf32>, vector<1x4xf32>
  %w = vector.transfer_write %v0, %arg0[%c0, %c0] :
    vector<1x4xf32>, tensor<4x4xf32>
  return
}

// -----

// CHECK-LABEL: func @dead_load
//   CHECK-NOT:   vector.maskedload
//   CHECK-NOT:   vector.gather
//   CHECK-NOT:   vector.expandload
//       CHECK:   return
func @dead_load(%base: memref<?xf32>, %indices: vector<16xi32>,
                          %mask: vector<16xi1>, %passthru: vector<16xf32>) {
  %c0 = constant 0 : index
  %0 = vector.maskedload %base[%c0], %mask, %passthru :
    memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %1 = vector.gather %base[%indices], %mask, %passthru :
    memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %2 = vector.expandload %base[%c0], %mask, %passthru :
    memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return
}

// -----

#contraction_accesses0 = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#contraction_trait0 = {
  indexing_maps = #contraction_accesses0,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @contractions
//  CHECK-SAME:   %[[A:[0-9a-zA-Z]+]]: vector<2x3xf32>
//  CHECK-SAME:   %[[B:[0-9a-zA-Z]+]]: vector<3x4xf32>
//  CHECK-SAME:   %[[C:[0-9a-zA-Z]+]]: vector<2x4xf32>
//  CHECK-SAME:   %[[A_I8:[0-9a-zA-Z]+]]: vector<2x3xi8>
//  CHECK-SAME:   %[[B_I8:[0-9a-zA-Z]+]]: vector<3x4xi8>
//  CHECK-SAME:   %[[C_I8:[0-9a-zA-Z]+]]: vector<2x4xi8>
func @contractions(%a: vector<2x3xf32>, %b: vector<3x4xf32>, %c: vector<2x4xf32>,
                   %a_i8: vector<2x3xi8>, %b_i8: vector<3x4xi8>, %c_i8: vector<2x4xi8>)
  -> (vector<2x4xf32>, vector<2x4xi8>)
{
  // CHECK-NOT: constant
  %vf_0 = constant dense <0.0>: vector<2x4xf32>
  // CHECK-NOT: addf
  //     CHECK: %[[D:.*]] = vector.contract {{.*}} %[[A]], %[[B]], %[[C]]
  %0 = vector.contract #contraction_trait0 %a, %b, %vf_0:
    vector<2x3xf32>, vector<3x4xf32> into vector<2x4xf32>
  // CHECK-NOT: addf
  %1 = addf %0, %c: vector<2x4xf32>

  // CHECK-NOT: constant
  %vi8_0 = constant dense <0>: vector<2x4xi8>
  // CHECK-NOT: addi
  //     CHECK: %[[D_I8:.*]] = vector.contract {{.*}} %[[A_I8]], %[[B_I8]], %[[C_I8]]
  %i8_0 = vector.contract #contraction_trait0 %a_i8, %b_i8, %vi8_0:
    vector<2x3xi8>, vector<3x4xi8> into vector<2x4xi8>
  // CHECK-NOT: addi
  %i8_1 = addi %i8_0, %c_i8: vector<2x4xi8>

  // CHECK: return %[[D]], %[[D_I8]]
  return %1, %i8_1: vector<2x4xf32>, vector<2x4xi8>
}

