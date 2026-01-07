// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

!vecA = vector<1x1x1xbf16>
!vecB = vector<1x1x8xbf16>
!vecC = vector<1x8xf32>
!memrefA = memref<1x4x1xbf16>
!memrefB = memref<1x1x32xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0,  d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0,  d1, d2, d3) -> (d1, d2)>
func.func @shuffle_VC_output_flat_layout(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = ub.poison : bf16
  %1 = vector.load %arg0[%c0, %c0, %c0] :
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0, %c0] :
        !memrefB, !vecB
  %3 = vector.load %arg1[%c0, %c0, %c8] :
        !memrefB, !vecB
  %4 = vector.load %arg2[%c0, %c0] :
        !memrefC, !vecC
  %5 = vector.load %arg2[%c0, %c8] :
        !memrefC, !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %4
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %5
    : !vecA, !vecB into !vecC

  vector.store %6, %arg2[%c0, %c0] : !memrefC, !vecC

  vector.store %7, %arg2[%c0, %c8] : !memrefC, !vecC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @shuffle_VC_output_flat_layout
// CHECK: vector.shuffle{{.*}}[0, 8, 1, 9, 2, 10, 3, 11] : vector<8xf32>, vector<8xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK: vector.contract
// CHECK: vector.shuffle{{.*}}[0, 8, 1, 9, 2, 10, 3, 11] : vector<8xf32>, vector<8xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.shuffle_bf16_vector_contract_result
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1xbf16>
!vecB = vector<1x8xbf16>
!vecC = vector<1x8xf32>
!memrefA = memref<4x1xbf16>
!memrefB = memref<1x32xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @shuffle_VC_output_flat_layout_transfer_read(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %3 = vector.transfer_read %arg1[%c0, %c8], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %4 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %5 = vector.transfer_read %arg2[%c0, %c8], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %4
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %5
    : !vecA, !vecB into !vecC

  vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  vector.transfer_write %7, %arg2[%c0, %c8] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @shuffle_VC_output_flat_layout_transfer_read
// CHECK: vector.shuffle{{.*}}[0, 8, 1, 9, 2, 10, 3, 11] : vector<8xf32>, vector<8xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK: vector.contract
// CHECK: vector.shuffle{{.*}}[0, 8, 1, 9, 2, 10, 3, 11] : vector<8xf32>, vector<8xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.shuffle_bf16_vector_contract_result
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
func.func @shuffle_VC_output_flat_layout_bf16dp(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %3 = vector.transfer_read %arg1[%c0, %c16], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %4 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %5 = vector.transfer_read %arg2[%c0, %c16], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %4
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %5
    : !vecA, !vecB into !vecC

  vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  vector.transfer_write %7, %arg2[%c0, %c16] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @shuffle_VC_output_flat_layout_bf16dp
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.contract
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.shuffle_bf16_vector_contract_result
    } : !transform.any_op
    transform.yield
  }
}
