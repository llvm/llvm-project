// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file -allow-unregistered-dialect | FileCheck %s

// This file contains tests for the vector.splat operation.
// Note that vector.splat is deprecated and will be removed.
// vector.broadcast should be used instead. These tests all
// have equivalent tests using vector.broadcast in canonicalize.mlir

// CHECK-LABEL: fold_extract_splat
//  CHECK-SAME:   %[[A:.*]]: f32
//       CHECK:   return %[[A]] : f32
func.func @fold_extract_splat(%a : f32, %idx0 : index, %idx1 : index, %idx2 : index) -> f32 {
  %b = vector.splat %a : vector<1x2x4xf32>
  %r = vector.extract %b[%idx0, %idx1, %idx2] : f32 from vector<1x2x4xf32>
  return %r : f32
}

// -----

// CHECK-LABEL: extract_strided_splat
//       CHECK:   %[[B:.*]] = vector.broadcast %{{.*}} f16 to vector<2x4xf16>
//  CHECK-NEXT:   return %[[B]] : vector<2x4xf16>
func.func @extract_strided_splat(%arg0: f16) -> vector<2x4xf16> {
 %0 = vector.splat %arg0 : vector<16x4xf16>
 %1 = vector.extract_strided_slice %0
  {offsets = [1, 0], sizes = [2, 4], strides = [1, 1]} :
  vector<16x4xf16> to vector<2x4xf16>
  return %1 : vector<2x4xf16>
}

// -----

// CHECK-LABEL: func @splat_fold
func.func @splat_fold() -> vector<4xf32> {
  %c = arith.constant 1.0 : f32
  %v = vector.splat %c : vector<4xf32>
  return %v : vector<4xf32>

  // CHECK-NEXT: [[V:%.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
  // CHECK-NEXT: return [[V]] : vector<4xf32>
}

// -----

// CHECK-LABEL: func @transpose_splat_constant
//       CHECK:   %[[CST:.+]] = arith.constant dense<5.000000e+00> : vector<8x4xf32>
//       CHECK:   return %[[CST]]
func.func @transpose_splat_constant() -> vector<8x4xf32> {
  %cst = arith.constant dense<5.0> : vector<4x8xf32>
  %0 = vector.transpose %cst, [1, 0] : vector<4x8xf32> to vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// CHECK-LABEL:   func @transpose_splat2(
// CHECK-SAME:                           %[[VAL_0:.*]]: f32) -> vector<3x4xf32> {
  // CHECK:           %[[VAL_1:.*]] = vector.broadcast %[[VAL_0]] : f32 to vector<3x4xf32>
// CHECK:           return %[[VAL_1]] : vector<3x4xf32>
// CHECK:         }
func.func @transpose_splat2(%arg : f32) -> vector<3x4xf32> {
  %splat = vector.broadcast %arg : f32 to vector<4x3xf32>
  %0 = vector.transpose %splat, [1, 0] : vector<4x3xf32> to vector<3x4xf32>
  return %0 : vector<3x4xf32>
}

// -----

// CHECK-LABEL: func @extract_element_splat_fold
//  CHECK-SAME: (%[[ARG:.+]]: i32)
//       CHECK:   return %[[ARG]]
func.func @extract_element_splat_fold(%a : i32) -> i32 {
  %v = vector.splat %a : vector<4xi32>
  %i = arith.constant 2 : i32
  %1 = vector.extractelement %v[%i : i32] : vector<4xi32>
  return %1 : i32
}

// -----

// CHECK-LABEL: @insert_strided_slice_splat
//  CHECK-SAME: (%[[ARG:.*]]: f32)
//  CHECK-NEXT:   %[[SPLAT:.*]] = vector.broadcast %[[ARG]] : f32 to vector<8x16xf32>
//  CHECK-NEXT:   return %[[SPLAT]] : vector<8x16xf32>
func.func @insert_strided_slice_splat(%x: f32) -> (vector<8x16xf32>) {
  %splat0 = vector.splat %x : vector<4x4xf32>
  %splat1 = vector.splat %x : vector<8x16xf32>
  %0 = vector.insert_strided_slice %splat0, %splat1 {offsets = [2, 2], strides = [1, 1]}
    : vector<4x4xf32> into vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: func @shuffle_splat
//  CHECK-SAME:   (%[[ARG:.*]]: i32)
//  CHECK-NEXT:   %[[SPLAT:.*]] = vector.broadcast %[[ARG]] : i32 to vector<4xi32>
//  CHECK-NEXT:   return %[[SPLAT]] : vector<4xi32>
func.func @shuffle_splat(%x : i32) -> vector<4xi32> {
  %v0 = vector.splat %x : vector<4xi32>
  %v1 = vector.splat %x : vector<2xi32>
  %shuffle = vector.shuffle %v0, %v1 [2, 3, 4, 5] : vector<4xi32>, vector<2xi32>
  return %shuffle : vector<4xi32>
}


// -----

// CHECK-LABEL: func @insert_splat
//  CHECK-SAME:   (%[[ARG:.*]]: i32)
//  CHECK-NEXT:   %[[SPLAT:.*]] = vector.broadcast %[[ARG]] : i32 to vector<2x4x3xi32>
//  CHECK-NEXT:   return %[[SPLAT]] : vector<2x4x3xi32>
func.func @insert_splat(%x : i32) -> vector<2x4x3xi32> {
  %v0 = vector.splat %x : vector<4x3xi32>
  %v1 = vector.splat %x : vector<2x4x3xi32>
  %insert = vector.insert %v0, %v1[0] : vector<4x3xi32> into vector<2x4x3xi32>
  return %insert : vector<2x4x3xi32>
}

// -----

// CHECK-LABEL: func @extract_from_0d_splat_broadcast_regression(
//  CHECK-SAME:     %[[a:.*]]: f32, %[[b:.*]]: vector<f32>, %[[c:.*]]: vector<2xf32>)
func.func @extract_from_0d_splat_broadcast_regression(%a: f32, %b: vector<f32>, %c: vector<2xf32>) -> (f32, f32, f32, f32, f32, vector<6x7xf32>, vector<3xf32>) {
  // Splat scalar to 0D and extract scalar.
  %0 = vector.splat %a : vector<f32>
  %1 = vector.extract %0[] : f32 from vector<f32>

  // Broadcast scalar to 0D and extract scalar.
  %2 = vector.broadcast %a : f32 to vector<f32>
  %3 = vector.extract %2[] : f32 from vector<f32>

  // Broadcast 0D to 3D and extract scalar.
  // CHECK: %[[extract1:.*]] = vector.extract %[[b]][] : f32 from vector<f32>
  %4 = vector.broadcast %b : vector<f32> to vector<1x2x4xf32>
  %5 = vector.extract %4[0, 0, 1] : f32 from vector<1x2x4xf32>

  // Splat scalar to 2D and extract scalar.
  %6 = vector.splat %a : vector<2x3xf32>
  %7 = vector.extract %6[0, 1] : f32 from vector<2x3xf32>

  // Broadcast scalar to 3D and extract scalar.
  %8 = vector.broadcast %a : f32 to vector<5x6x7xf32>
  %9 = vector.extract %8[2, 1, 5] : f32 from vector<5x6x7xf32>

  // Extract 2D from 3D that was broadcasted from a scalar.
  // CHECK: %[[extract2:.*]] = vector.broadcast %[[a]] : f32 to vector<6x7xf32>
  %10 = vector.extract %8[2] : vector<6x7xf32> from vector<5x6x7xf32>

  // Extract 1D from 2D that was splat'ed from a scalar.
  // CHECK: %[[extract3:.*]] = vector.broadcast %[[a]] : f32 to vector<3xf32>
  %11 = vector.extract %6[1] : vector<3xf32> from vector<2x3xf32>

  // CHECK:   return %[[a]], %[[a]], %[[extract1]], %[[a]], %[[a]], %[[extract2]], %[[extract3]]
  return %1, %3, %5, %7, %9, %10, %11 : f32, f32, f32, f32, f32, vector<6x7xf32>, vector<3xf32>
}
