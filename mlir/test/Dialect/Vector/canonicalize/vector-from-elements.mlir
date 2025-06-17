// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file -allow-unregistered-dialect | FileCheck %s

// This file contains some tests of folding/canonicalizing vector.from_elements

///===----------------------------------------------===//
///  Tests of `rewriteFromElementsAsSplat`
///===----------------------------------------------===//

// CHECK-LABEL: func @extract_scalar_from_from_elements(
//  CHECK-SAME:      %[[A:.*]]: f32, %[[B:.*]]: f32)
func.func @extract_scalar_from_from_elements(%a: f32, %b: f32) -> (f32, f32, f32, f32, f32, f32, f32) {
  // Extract from 0D.
  %0 = vector.from_elements %a : vector<f32>
  %1 = vector.extract %0[] : f32 from vector<f32>

  // Extract from 1D.
  %2 = vector.from_elements %a : vector<1xf32>
  %3 = vector.extract %2[0] : f32 from vector<1xf32>
  %4 = vector.from_elements %a, %b, %a, %a, %b : vector<5xf32>
  %5 = vector.extract %4[4] : f32 from vector<5xf32>

  // Extract from 2D.
  %6 = vector.from_elements %a, %a, %a, %b, %b, %b : vector<2x3xf32>
  %7 = vector.extract %6[0, 0] : f32 from vector<2x3xf32>
  %8 = vector.extract %6[0, 1] : f32 from vector<2x3xf32>
  %9 = vector.extract %6[1, 1] : f32 from vector<2x3xf32>
  %10 = vector.extract %6[1, 2] : f32 from vector<2x3xf32>

  // CHECK: return %[[A]], %[[A]], %[[B]], %[[A]], %[[A]], %[[B]], %[[B]]
  return %1, %3, %5, %7, %8, %9, %10 : f32, f32, f32, f32, f32, f32, f32
}

// -----

// CHECK-LABEL: func @extract_1d_from_from_elements(
//  CHECK-SAME:      %[[A:.*]]: f32, %[[B:.*]]: f32)
func.func @extract_1d_from_from_elements(%a: f32, %b: f32) -> (vector<3xf32>, vector<3xf32>) {
  %0 = vector.from_elements %a, %a, %a, %b, %b, %b : vector<2x3xf32>
  // CHECK: %[[SPLAT1:.*]] = vector.splat %[[A]] : vector<3xf32>
  %1 = vector.extract %0[0] : vector<3xf32> from vector<2x3xf32>
  // CHECK: %[[SPLAT2:.*]] = vector.splat %[[B]] : vector<3xf32>
  %2 = vector.extract %0[1] : vector<3xf32> from vector<2x3xf32>
  // CHECK: return %[[SPLAT1]], %[[SPLAT2]]
  return %1, %2 : vector<3xf32>, vector<3xf32>
}

// -----

// CHECK-LABEL: func @extract_2d_from_from_elements(
//  CHECK-SAME:      %[[A:.*]]: f32, %[[B:.*]]: f32)
func.func @extract_2d_from_from_elements(%a: f32, %b: f32) -> (vector<2x2xf32>, vector<2x2xf32>) {
  %0 = vector.from_elements %a, %a, %a, %b, %b, %b, %b, %a, %b, %a, %a, %b : vector<3x2x2xf32>
  // CHECK: %[[SPLAT1:.*]] = vector.from_elements %[[A]], %[[A]], %[[A]], %[[B]] : vector<2x2xf32>
  %1 = vector.extract %0[0] : vector<2x2xf32> from vector<3x2x2xf32>
  // CHECK: %[[SPLAT2:.*]] = vector.from_elements %[[B]], %[[B]], %[[B]], %[[A]] : vector<2x2xf32>
  %2 = vector.extract %0[1] : vector<2x2xf32> from vector<3x2x2xf32>
  // CHECK: return %[[SPLAT1]], %[[SPLAT2]]
  return %1, %2 : vector<2x2xf32>, vector<2x2xf32>
}

// -----

// CHECK-LABEL: func @from_elements_to_splat(
//  CHECK-SAME:      %[[A:.*]]: f32, %[[B:.*]]: f32)
func.func @from_elements_to_splat(%a: f32, %b: f32) -> (vector<2x3xf32>, vector<2x3xf32>, vector<f32>) {
  // CHECK: %[[SPLAT:.*]] = vector.splat %[[A]] : vector<2x3xf32>
  %0 = vector.from_elements %a, %a, %a, %a, %a, %a : vector<2x3xf32>
  // CHECK: %[[FROM_EL:.*]] = vector.from_elements {{.*}} : vector<2x3xf32>
  %1 = vector.from_elements %a, %a, %a, %a, %b, %a : vector<2x3xf32>
  // CHECK: %[[SPLAT2:.*]] = vector.splat %[[A]] : vector<f32>
  %2 = vector.from_elements %a : vector<f32>
  // CHECK: return %[[SPLAT]], %[[FROM_EL]], %[[SPLAT2]]
  return %0, %1, %2 : vector<2x3xf32>, vector<2x3xf32>, vector<f32>
}

// -----

///===----------------------------------------------===//
///  Tests of `FromElementsToShapeCast`
///===----------------------------------------------===//

// CHECK-LABEL: func @to_shape_cast_rank2_to_rank1(
//  CHECK-SAME:       %[[A:.*]]: vector<1x2xi8>)
//       CHECK:       %[[EXTRACT:.*]] = vector.extract %[[A]][0] : vector<2xi8> from vector<1x2xi8>
//       CHECK:       return %[[EXTRACT]] : vector<2xi8>
func.func @to_shape_cast_rank2_to_rank1(%arg0: vector<1x2xi8>) -> vector<2xi8> {
  %0 = vector.extract %arg0[0, 0] : i8 from vector<1x2xi8>
  %1 = vector.extract %arg0[0, 1] : i8 from vector<1x2xi8>
  %4 = vector.from_elements %0, %1 : vector<2xi8>
  return %4 : vector<2xi8>
}

// -----

// CHECK-LABEL: func @to_shape_cast_rank1_to_rank3(
//  CHECK-SAME:       %[[A:.*]]: vector<8xi8>)
//       CHECK:       %[[SHAPE_CAST:.*]] = vector.shape_cast %[[A]] : vector<8xi8> to vector<2x2x2xi8>
//       CHECK:       return %[[SHAPE_CAST]] : vector<2x2x2xi8>
func.func @to_shape_cast_rank1_to_rank3(%arg0: vector<8xi8>) -> vector<2x2x2xi8> {
  %0 = vector.extract %arg0[0] : i8 from vector<8xi8>
  %1 = vector.extract %arg0[1] : i8 from vector<8xi8>
  %2 = vector.extract %arg0[2] : i8 from vector<8xi8>
  %3 = vector.extract %arg0[3] : i8 from vector<8xi8>
  %4 = vector.extract %arg0[4] : i8 from vector<8xi8>
  %5 = vector.extract %arg0[5] : i8 from vector<8xi8>
  %6 = vector.extract %arg0[6] : i8 from vector<8xi8>
  %7 = vector.extract %arg0[7] : i8 from vector<8xi8>
  %8 = vector.from_elements %0, %1, %2, %3, %4, %5, %6, %7 : vector<2x2x2xi8>
  return %8 : vector<2x2x2xi8>
}

// -----

// CHECK-LABEL: func @source_larger_than_out(
//  CHECK-SAME:      %[[A:.*]]: vector<2x3x4xi8>)
//       CHECK:      %[[EXTRACT:.*]] = vector.extract %[[A]][1] : vector<3x4xi8> from vector<2x3x4xi8>
//       CHECK:      %[[SHAPE_CAST:.*]] = vector.shape_cast %[[EXTRACT]] : vector<3x4xi8> to vector<12xi8>
//       CHECK:      return %[[SHAPE_CAST]] : vector<12xi8>
func.func @source_larger_than_out(%arg0: vector<2x3x4xi8>) -> vector<12xi8> {
  %0 = vector.extract %arg0[1, 0, 0] : i8 from vector<2x3x4xi8>
  %1 = vector.extract %arg0[1, 0, 1] : i8 from vector<2x3x4xi8>
  %2 = vector.extract %arg0[1, 0, 2] : i8 from vector<2x3x4xi8>
  %3 = vector.extract %arg0[1, 0, 3] : i8 from vector<2x3x4xi8>
  %4 = vector.extract %arg0[1, 1, 0] : i8 from vector<2x3x4xi8>
  %5 = vector.extract %arg0[1, 1, 1] : i8 from vector<2x3x4xi8>
  %6 = vector.extract %arg0[1, 1, 2] : i8 from vector<2x3x4xi8>
  %7 = vector.extract %arg0[1, 1, 3] : i8 from vector<2x3x4xi8>
  %8 = vector.extract %arg0[1, 2, 0] : i8 from vector<2x3x4xi8>
  %9 = vector.extract %arg0[1, 2, 1] : i8 from vector<2x3x4xi8>
  %10 = vector.extract %arg0[1, 2, 2] : i8 from vector<2x3x4xi8>
  %11 = vector.extract %arg0[1, 2, 3] : i8 from vector<2x3x4xi8>
  %12 = vector.from_elements %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11 : vector<12xi8>
  return %12 : vector<12xi8>
}

// -----

// This test is similar to `source_larger_than_out` except here the number of elements
// extracted contigously starting from the first position [0,0] could be 6 instead of 3
// and the pattern would still match.
// CHECK-LABEL: func @suffix_with_excess_zeros(
//       CHECK:      %[[EXT:.*]] = vector.extract {{.*}}[0] : vector<3xi8> from vector<2x3xi8>
//       CHECK:      return %[[EXT]] : vector<3xi8>
func.func @suffix_with_excess_zeros(%arg0: vector<2x3xi8>) -> vector<3xi8> {
  %0 = vector.extract %arg0[0, 0] : i8 from vector<2x3xi8>
  %1 = vector.extract %arg0[0, 1] : i8 from vector<2x3xi8>
  %2 = vector.extract %arg0[0, 2] : i8 from vector<2x3xi8>
  %3 = vector.from_elements %0, %1, %2 : vector<3xi8>
  return %3 : vector<3xi8>
}

// -----

// CHECK-LABEL: func @large_source_with_shape_cast_required(
//  CHECK-SAME:      %[[A:.*]]: vector<2x2x2x2xi8>)
//       CHECK:      %[[EXTRACT:.*]] = vector.extract %[[A]][0, 1] : vector<2x2xi8> from vector<2x2x2x2xi8>
//       CHECK:      %[[SHAPE_CAST:.*]] = vector.shape_cast %[[EXTRACT]] : vector<2x2xi8> to vector<1x4x1xi8>
//       CHECK:      return %[[SHAPE_CAST]] : vector<1x4x1xi8>
func.func @large_source_with_shape_cast_required(%arg0: vector<2x2x2x2xi8>) -> vector<1x4x1xi8> {
  %0 = vector.extract %arg0[0, 1, 0, 0] : i8 from vector<2x2x2x2xi8>
  %1 = vector.extract %arg0[0, 1, 0, 1] : i8 from vector<2x2x2x2xi8>
  %2 = vector.extract %arg0[0, 1, 1, 0] : i8 from vector<2x2x2x2xi8>
  %3 = vector.extract %arg0[0, 1, 1, 1] : i8 from vector<2x2x2x2xi8>
  %4 = vector.from_elements %0, %1, %2, %3 : vector<1x4x1xi8>
  return %4 : vector<1x4x1xi8>
}

//  -----

// Could match, but handled by `rewriteFromElementsAsSplat`.
// CHECK-LABEL: func @extract_single_elm(
//  CHECK-NEXT:      vector.extract
//  CHECK-NEXT:      vector.splat
//  CHECK-NEXT:      return
func.func @extract_single_elm(%arg0 : vector<2x3xi8>) -> vector<1xi8> {
  %0 = vector.extract %arg0[0, 0] : i8 from vector<2x3xi8>
  %1 = vector.from_elements %0 : vector<1xi8>
  return %1 : vector<1xi8>
}

// -----

//   CHECK-LABEL: func @negative_source_contiguous_but_not_suffix(
//     CHECK-NOT:      shape_cast
//         CHECK:      from_elements
func.func @negative_source_contiguous_but_not_suffix(%arg0: vector<2x3xi8>) -> vector<3xi8> {
  %0 = vector.extract %arg0[0, 1] : i8 from vector<2x3xi8>
  %1 = vector.extract %arg0[0, 2] : i8 from vector<2x3xi8>
  %2 = vector.extract %arg0[1, 0] : i8 from vector<2x3xi8>
  %3 = vector.from_elements %0, %1, %2 : vector<3xi8>
  return %3 : vector<3xi8>
}

// -----

// The extracted elements are recombined into a single vector, but in a new order.
// CHECK-LABEL: func @negative_nonascending_order(
//   CHECK-NOT:      shape_cast
//       CHECK:      from_elements
func.func @negative_nonascending_order(%arg0: vector<1x2xi8>) -> vector<2xi8> {
  %0 = vector.extract %arg0[0, 1] : i8 from vector<1x2xi8>
  %1 = vector.extract %arg0[0, 0] : i8 from vector<1x2xi8>
  %2 = vector.from_elements %0, %1 : vector<2xi8>
  return %2 : vector<2xi8>
}

// -----

// CHECK-LABEL: func @negative_nonstatic_extract(
//   CHECK-NOT:      shape_cast
//       CHECK:      from_elements
func.func @negative_nonstatic_extract(%arg0: vector<1x2xi8>, %i0 : index, %i1 : index) -> vector<2xi8> {
  %0 = vector.extract %arg0[0, %i0] : i8 from vector<1x2xi8>
  %1 = vector.extract %arg0[0, %i1] : i8 from vector<1x2xi8>
  %2 = vector.from_elements %0, %1 : vector<2xi8>
  return %2 : vector<2xi8>
}

// -----

// CHECK-LABEL: func @negative_different_sources(
//   CHECK-NOT:      shape_cast
//       CHECK:      from_elements
func.func @negative_different_sources(%arg0: vector<1x2xi8>, %arg1: vector<1x2xi8>) -> vector<2xi8> {
  %0 = vector.extract %arg0[0, 0] : i8 from vector<1x2xi8>
  %1 = vector.extract %arg1[0, 1] : i8 from vector<1x2xi8>
  %2 = vector.from_elements %0, %1 : vector<2xi8>
  return %2 : vector<2xi8>
}

// -----

// CHECK-LABEL: func @negative_source_not_suffix(
//   CHECK-NOT:      shape_cast
//       CHECK:      from_elements
func.func @negative_source_not_suffix(%arg0: vector<1x3xi8>) -> vector<2xi8> {
  %0 = vector.extract %arg0[0, 0] : i8 from vector<1x3xi8>
  %1 = vector.extract %arg0[0, 1] : i8 from vector<1x3xi8>
  %2 = vector.from_elements %0, %1 : vector<2xi8>
  return %2 : vector<2xi8>
}

// -----

// The inserted elements are a subset of the extracted elements.
// [0, 1, 2] -> [1, 1, 2]
// CHECK-LABEL: func @negative_nobijection_order(
//   CHECK-NOT:      shape_cast
//       CHECK:      from_elements
func.func @negative_nobijection_order(%arg0: vector<1x3xi8>) -> vector<3xi8> {
  %0 = vector.extract %arg0[0, 1] : i8 from vector<1x3xi8>
  %1 = vector.extract %arg0[0, 2] : i8 from vector<1x3xi8>
  %2 = vector.from_elements %0, %0, %1 : vector<3xi8>
  return %2 : vector<3xi8>
}

// -----

// CHECK-LABEL: func @negative_source_too_small(
//   CHECK-NOT:      shape_cast
//       CHECK:      from_elements
func.func @negative_source_too_small(%arg0: vector<2xi8>) -> vector<4xi8> {
  %0 = vector.extract %arg0[0] : i8 from vector<2xi8>
  %1 = vector.extract %arg0[1] : i8 from vector<2xi8>
  %2 = vector.from_elements %0, %1, %1, %1 : vector<4xi8>
  return %2 : vector<4xi8>
}

