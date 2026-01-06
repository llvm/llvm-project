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
// CHECK: vector.shuffle
// CHECK-NEXT: vector.shuffle
// CHECK: vector.contract
// CHECK: vector.shuffle
// CHECK-NEXT: vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.shuffle_bf16_vector_contract_result
    } : !transform.any_op
    transform.yield
  }
}
