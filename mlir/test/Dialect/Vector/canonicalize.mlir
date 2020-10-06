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
