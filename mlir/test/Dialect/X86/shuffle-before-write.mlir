// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

// Keep these cases independent of the move-accumulator patterns exercised by
// the larger contraction tests. They specifically cover listener notification
// when shuffleBeforeWriteLikeOp updates two write-like users in place.

!vecA = vector<1x1xbf16>
!vecB = vector<1x8xbf16>
!vecC = vector<1x8xf32>
!memrefA = memref<4x1xbf16>
!memrefB = memref<1x32xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>

func.func @matmul_to_fma_flat_layout(
    %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %poisonBf16 = ub.poison : bf16
  %poisonF32 = ub.poison : f32
  %lhs = vector.transfer_read %arg0[%c0, %c0], %poisonBf16
      {in_bounds = [true, true]} : !memrefA, !vecA
  %rhs0 = vector.transfer_read %arg1[%c0, %c0], %poisonBf16
      {in_bounds = [true, true]} : !memrefB, !vecB
  %rhs1 = vector.transfer_read %arg1[%c0, %c8], %poisonBf16
      {in_bounds = [true, true]} : !memrefB, !vecB
  %acc0 = vector.transfer_read %arg2[%c0, %c0], %poisonF32
      {in_bounds = [true, true]} : !memrefC, !vecC
  %acc1 = vector.transfer_read %arg2[%c0, %c8], %poisonF32
      {in_bounds = [true, true]} : !memrefC, !vecC

  %result0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %lhs, %rhs0, %acc0 : !vecA, !vecB into !vecC
  %result1 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %lhs, %rhs1, %acc1 : !vecA, !vecB into !vecC

  vector.transfer_write %result0, %arg2[%c0, %c0]
      {in_bounds = [true, true]} : !vecC, !memrefC
  vector.transfer_write %result1, %arg2[%c0, %c8]
      {in_bounds = [true, true]} : !vecC, !memrefC
  return %arg2 : !memrefC
}

// CHECK-LABEL: @matmul_to_fma_flat_layout
// CHECK: vector.fma
// CHECK: vector.fma
// CHECK: %[[BF16_RESULT_LO:.*]] = vector.shuffle
// CHECK-NEXT: %[[BF16_RESULT_HI:.*]] = vector.shuffle
// CHECK-NEXT: %[[BF16_WRITE_LO:.*]] = vector.shape_cast %[[BF16_RESULT_LO]]
// CHECK-NEXT: %[[BF16_WRITE_HI:.*]] = vector.shape_cast %[[BF16_RESULT_HI]]
// CHECK-NEXT: vector.transfer_write %[[BF16_WRITE_LO]], %arg2[%c0, %c0]
// CHECK-NEXT: vector.transfer_write %[[BF16_WRITE_HI]], %arg2[%c0, %c8]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_bf16_to_fma
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x2xbf16>
!vecB = vector<2x16xbf16>
!vecC = vector<1x16xf32>
!memrefA = memref<4x2xbf16>
!memrefB = memref<2x32xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>

func.func @matmul_bf16dp_flat_layout_B_shuffled(
    %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %poisonBf16 = ub.poison : bf16
  %poisonF32 = ub.poison : f32
  %lhs = vector.load %arg0[%c0, %c0] : !memrefA, !vecA
  %rhs0 = vector.load %arg1[%c0, %c0] : !memrefB, !vecB
  %rhs1 = vector.load %arg1[%c0, %c16] : !memrefB, !vecB
  %acc0 = vector.load %arg2[%c0, %c0] : !memrefC, !vecC
  %acc1 = vector.load %arg2[%c0, %c16] : !memrefC, !vecC

  %flatRhs0 = vector.shape_cast %rhs0 : !vecB to vector<32xbf16>
  %flatRhs1 = vector.shape_cast %rhs1 : !vecB to vector<32xbf16>
  %shuffle0 = vector.shuffle %flatRhs0, %flatRhs1
      [0, 32, 1, 33, 2, 34, 3, 35, 8, 40, 9, 41, 10, 42, 11, 43,
       16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59]
      : vector<32xbf16>, vector<32xbf16>
  %shuffle1 = vector.shuffle %flatRhs0, %flatRhs1
      [4, 36, 5, 37, 6, 38, 7, 39, 12, 44, 13, 45, 14, 46, 15, 47,
       20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63]
      : vector<32xbf16>, vector<32xbf16>
  %packedRhs0 = vector.shape_cast %shuffle0 : vector<32xbf16> to !vecB
  %packedRhs1 = vector.shape_cast %shuffle1 : vector<32xbf16> to !vecB

  %result0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %lhs, %packedRhs0, %acc0 : !vecA, !vecB into !vecC
  %result1 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %lhs, %packedRhs1, %acc1 : !vecA, !vecB into !vecC

  vector.store %result0, %arg2[%c0, %c0] : !memrefC, !vecC
  vector.store %result1, %arg2[%c0, %c16] : !memrefC, !vecC
  return %arg2 : !memrefC
}

// CHECK-LABEL: @matmul_bf16dp_flat_layout_B_shuffled
// CHECK: x86.avx512.dot
// CHECK: x86.avx512.dot
// CHECK: %[[PACKED_RESULT_LO:.*]] = vector.shuffle
// CHECK-NEXT: %[[PACKED_RESULT_HI:.*]] = vector.shuffle
// CHECK-NEXT: %[[PACKED_WRITE_LO:.*]] = vector.shape_cast %[[PACKED_RESULT_LO]]
// CHECK-NEXT: %[[PACKED_WRITE_HI:.*]] = vector.shape_cast %[[PACKED_RESULT_HI]]
// CHECK-NEXT: vector.store %[[PACKED_WRITE_LO]], %arg2[%c0, %c0]
// CHECK-NEXT: vector.store %[[PACKED_WRITE_HI]], %arg2[%c0, %c16]
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}
