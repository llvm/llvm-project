// RUN: mlir-opt %s -split-input-file -canonicalize |  FileCheck %s

// This file contains tests where a vector.shape_cast is the result
// of canonicalization.

// **--------------------------------------------------------** //
//   Tests of BroadcastToShapeCast
// **--------------------------------------------------------** //

// CHECK-LABEL: @broadcast_to_shape_cast
//  CHECK-SAME: %[[ARG0:.*]]: vector<4xi8>
//  CHECK-NEXT: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]]
//  CHECK-NEXT: return %[[SHAPE_CAST]] : vector<1x1x4xi8>
func.func @broadcast_to_shape_cast(%arg0 : vector<4xi8>) -> vector<1x1x4xi8> {
  %0 = vector.broadcast %arg0 : vector<4xi8> to vector<1x1x4xi8>
  return %0 : vector<1x1x4xi8>
}

// -----

// broadcast can only be transformed to a shape_cast if the number of elements is
// unchanged by the broadcast.
// CHECK-LABEL: @negative_broadcast_increased_elements_to_shape_cast
//   CHECK-NOT: shape_cast
//       CHECK: return
func.func @negative_broadcast_increased_elements_to_shape_cast(%arg0 : vector<1x4xi8>) -> vector<2x3x4xi8> {
  %0 = vector.broadcast %arg0 : vector<1x4xi8> to vector<2x3x4xi8>
  return %0 : vector<2x3x4xi8>
}

// -----

// shape_cast does not support scalar inputs/outputs, so a broadcast of a scalar
// cannot be transformed to a shape_cast.
// CHECK-LABEL: @negative_broadcast_scalar_to_shape_cast
//   CHECK-NOT: shape_cast
//       CHECK: return
func.func @negative_broadcast_scalar_to_shape_cast(%arg0 : i8) -> vector<1xi8> {
  %0 = vector.broadcast %arg0 : i8 to vector<1xi8>
  return %0 : vector<1xi8>
}

// -----

// In this test, broadcast (2)->(1,2,1) is not legal, but shape_cast (2)->(1,2,1) is.
// CHECK-LABEL: func @canonicalize_broadcast_shapecast_to_shapecast
//   CHECK-NOT:   vector.broadcast
//       CHECK:   vector.shape_cast {{.+}} : vector<2xf32> to vector<1x2x1xf32>
func.func @canonicalize_broadcast_shapecast_to_shapecast(%arg0 : vector<2xf32>) -> vector<1x2x1xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<1x2xf32>
  %1 = vector.shape_cast %0 : vector<1x2xf32> to vector<1x2x1xf32>
  return %1 : vector<1x2x1xf32>
}

// -----

// In this test, broadcast (1)->(1,1) and shape_cast (1)->(1,1) are both legal. shape_cast is chosen.
// CHECK-LABEL: func @canonicalize_broadcast_shapecast_both_possible
//   CHECK-NOT:   vector.broadcast
//       CHECK:   vector.shape_cast {{.+}} : vector<1xf32> to vector<1x1xf32>
func.func @canonicalize_broadcast_shapecast_both_possible(%arg0: vector<1xf32>) -> vector<1x1xf32> {
    %0 = vector.broadcast %arg0 : vector<1xf32> to vector<1x1x1xf32>
    %1 = vector.shape_cast %0 : vector<1x1x1xf32> to vector<1x1xf32>
    return %1 : vector<1x1xf32>
}


// -----

// **--------------------------------------------------------** //
//   Tests of TransposeToShapeCast
// **--------------------------------------------------------** //

// In this test, the permutation maps the non-unit dimensions (0 and 2) are as follows:
// 0 -> 0
// 2 -> 1
// Because 0 < 1, this permutation is order preserving and effectively a shape_cast.
// shape_cast is canonical form of all reshapes, so check that this transpose is
// transformed to a shape_cast.
// CHECK-LABEL: @transpose_to_shape_cast
//  CHECK-SAME: %[[ARG0:.*]]: vector<2x1x2xf32>
//  CHECK-NEXT: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]]
//  CHECK-NEXT: return %[[SHAPE_CAST]] : vector<2x2x1xf32>
func.func @transpose_to_shape_cast(%arg0 : vector<2x1x2xf32>) -> vector<2x2x1xf32> {
  %0 = vector.transpose %arg0, [0, 2, 1] : vector<2x1x2xf32> to vector<2x2x1xf32>
  return %0 : vector<2x2x1xf32>
}

// -----

// In this test, the permutation maps the non-unit dimensions (1 and 2) as follows:
// 1 -> 0
// 2 -> 4
// Because 0 < 4, this permutation is order preserving, and therefore we expect it
// to be converted to a shape_cast.
// CHECK-LABEL: @shape_cast_of_transpose
//  CHECK-SAME: %[[ARG:.*]]: vector<1x4x4x1x1xi8>)
//       CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG]] :
//  CHECK-SAME: vector<1x4x4x1x1xi8> to vector<4x1x1x1x4xi8>
//       CHECK: return %[[SHAPE_CAST]]
func.func @shape_cast_of_transpose(%arg : vector<1x4x4x1x1xi8>) -> vector<4x1x1x1x4xi8> {
  %0 = vector.transpose %arg, [1, 0, 3, 4, 2]  : vector<1x4x4x1x1xi8> to vector<4x1x1x1x4xi8>
  return %0 : vector<4x1x1x1x4xi8>
}

// -----

// Scalable dimensions should be treated as non-unit dimensions.
// CHECK-LABEL: @transpose_scalable_unit
//   CHECK-NOT: shape_cast
func.func @transpose_scalable_unit(%arg : vector<[1]x4xi8>) -> vector<4x[1]xi8> {
  %0 = vector.transpose %arg, [1, 0] : vector<[1]x4xi8> to vector<4x[1]xi8>
  return %0 : vector<4x[1]xi8>
}

// -----

// In this test, the mapping of non-unit dimensions (1 and 2) is as follows:
// 1 -> 2
// 2 -> 1
// As this is not increasing (2 > 1), this transpose is not order
// preserving and cannot be treated as a shape_cast.
// CHECK-LABEL: @negative_transpose_to_shape_cast
//   CHECK-NOT: shape_cast
func.func @negative_transpose_to_shape_cast(%arg : vector<1x4x4x1xi8>) -> vector<1x4x4x1xi8> {
  %0 = vector.transpose %arg, [0, 2, 1, 3]
     : vector<1x4x4x1xi8> to vector<1x4x4x1xi8>
  return %0 : vector<1x4x4x1xi8>
}

// -----

// CHECK-LABEL: @shape_cast_of_transpose_scalable
//  CHECK-NEXT: vector.shape_cast
//  CHECK-NEXT: return
func.func @shape_cast_of_transpose_scalable(%arg : vector<[4]x1xi8>) -> vector<[4]xi8> {
  %0 = vector.transpose %arg, [1, 0] : vector<[4]x1xi8> to vector<1x[4]xi8>
  %1 = vector.shape_cast %0 : vector<1x[4]xi8> to vector<[4]xi8>
  return %1 : vector<[4]xi8>
}

// -----

// CHECK-LABEL: @transpose_of_shape_cast_scalable
//  CHECK-NEXT: vector.shape_cast
//  CHECK-NEXT: return
func.func @transpose_of_shape_cast_scalable(%arg : vector<[4]xi8>) -> vector<[4]x1xi8> {
  %0 = vector.shape_cast %arg : vector<[4]xi8> to vector<1x[4]xi8>
  %1 = vector.transpose %0, [1, 0] : vector<1x[4]xi8> to vector<[4]x1xi8>
  return %1 : vector<[4]x1xi8>
}

// -----

// A test where a transpose cannot be transformed to a shape_cast because it is not order
// preserving
// CHECK-LABEL: @negative_transpose_to_shape_cast
//  CHECK-SAME: %[[ARG0:.*]]: vector<2x1x2xf32>
//  CHECK-NEXT: %[[TRANSPOSE:.*]] = vector.transpose %[[ARG0]], [2, 0, 1]
//  CHECK-NEXT: return %[[TRANSPOSE]] : vector<2x2x1xf32>
func.func @negative_transpose_to_shape_cast(%arg0 : vector<2x1x2xf32>) -> vector<2x2x1xf32> {
  %0 = vector.transpose %arg0, [2, 0, 1] : vector<2x1x2xf32> to vector<2x2x1xf32>
  return %0 : vector<2x2x1xf32>
}

// -----

// **--------------------------------------------------------** //
//   Tests of ExtractToShapeCast
// **--------------------------------------------------------** //

// CHECK-LABEL: @extract_to_shape_cast
//  CHECK-SAME: %[[ARG0:.*]]: vector<1x4xf32>
//  CHECK-NEXT: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]]
//  CHECK-NEXT: return %[[SHAPE_CAST]] : vector<4xf32>
func.func @extract_to_shape_cast(%arg0 : vector<1x4xf32>) -> vector<4xf32> {
  %0 = vector.extract %arg0[0] : vector<4xf32> from vector<1x4xf32>
  return %0 : vector<4xf32>
}

// -----

// In this example, arg1 might be negative indicating poison. We could
// convert this to shape_cast (would be a legal transform with poison)
// but we conservatively choose not to.
// CHECK-LABEL: @negative_extract_to_shape_cast
//   CHECK-NOT: shape_cast
func.func @negative_extract_to_shape_cast(%arg0 : vector<1x4xf32>, %arg1 : index) -> vector<4xf32> {
  %0 = vector.extract %arg0[%arg1] : vector<4xf32> from vector<1x4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: fold_extract_shapecast_to_shapecast
//  CHECK-SAME: (%[[ARG:.+]]: vector<3x4xf32>)
//       CHECK:   %[[R:.+]] = vector.shape_cast %[[ARG]] : vector<3x4xf32> to vector<12xf32>
//       CHECK:   return %[[R]]
func.func @fold_extract_shapecast_to_shapecast(%arg0 : vector<3x4xf32>) -> vector<12xf32> {
  %0 = vector.shape_cast %arg0 : vector<3x4xf32> to vector<1x12xf32>
  %r = vector.extract %0[0] : vector<12xf32> from vector<1x12xf32>
  return %r : vector<12xf32>
}

// -----

// CHECK-LABEL: func @insert_extract_to_shape_cast
//  CHECK-SAME: (%[[ARG0:.*]]: vector<1x1x4xf32>, %[[ARG1:.*]]: vector<4xf32>)
//       CHECK:   %[[V0:.*]] = vector.shape_cast %[[ARG0]] : vector<1x1x4xf32> to vector<4xf32>
//       CHECK:   %[[V1:.*]] = vector.shape_cast %[[ARG1]] : vector<4xf32> to vector<1x1x4xf32>
//       CHECK:   return %[[V0]], %[[V1]] : vector<4xf32>, vector<1x1x4xf32>
func.func @insert_extract_to_shape_cast(%arg0 : vector<1x1x4xf32>,
  %arg1 : vector<4xf32>) -> (vector<4xf32>, vector<1x1x4xf32>) {
  %0 = vector.extract %arg0[0, 0] : vector<4xf32> from vector<1x1x4xf32>
  %1 = vector.insert %arg1, %arg0 [0, 0] : vector<4xf32> into vector<1x1x4xf32>
  return %0, %1 : vector<4xf32>, vector<1x1x4xf32>
}

// -----

// CHECK-LABEL: func.func @extract_from_broadcast
func.func @extract_from_broadcast(%src: vector<1x1x1xf32>) -> vector<1xf32> {
  %0 = vector.broadcast %src : vector<1x1x1xf32> to vector<1x1x32x1xf32>
  //  CHECK-NEXT:   %[[RES:.*]] = vector.shape_cast{{.*}} vector<1x1x1xf32> to vector<1xf32>
  //  CHECK-NEXT:   return %[[RES]] : vector<1xf32>
  %1 = vector.extract %0[0, 0, 31] : vector<1xf32> from vector<1x1x32x1xf32>
  return %1: vector<1xf32>
}

// -----

// **--------------------------------------------------------** //
//   Tests of FromElementsToShapeCast
// **--------------------------------------------------------** //

// CHECK-LABEL: func @to_shape_cast_rank2_to_rank1(
//  CHECK-SAME:       %[[A:.*]]: vector<1x2xi8>)
//       CHECK:       %[[SC:.*]] = vector.shape_cast %[[A]] : vector<1x2xi8> to vector<2xi8>
//       CHECK:       return %[[SC]] : vector<2xi8>
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
//  CHECK-NEXT:      vector.broadcast
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

