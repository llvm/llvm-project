// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

!vecA = vector<1x1x1x2xbf16>
!vecB = vector<1x1x8x2xbf16>
!vecC = vector<1x8xf32>
!memrefA = memref<1x4x1x2xbf16>
!memrefB = memref<1x1x32x2xbf16>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @brgemm_to_fma(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !vecC) -> !vecC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %1 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : 
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefB, !vecB
  %3 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %arg2
    : !vecA, !vecB into !vecC
  return %3 : !vecC
}

// CHECK-LABEL: @brgemm_to_fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.vector_contract_bf16_to_fma
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1x2xbf16>
!vecB = vector<1x1x8x2xbf16>
!vecC = vector<1x8xf32>
!memrefA = memref<1x4x1x2xbf16>
!memrefB = memref<1x1x32x2xbf16>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @brgemm_to_fma_load(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !vecC) -> !vecC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %1 = vector.load %arg0[%c0, %c0, %c0, %c0] : 
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0, %c0, %c0] :
        !memrefB, !vecB
  %3 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %arg2
    : !vecA, !vecB into !vecC
  return %3 : !vecC
}

// CHECK-LABEL: @brgemm_to_fma_load
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.vector_contract_bf16_to_fma
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x8x1x2xbf16>
!vecB = vector<1x1x1x2xbf16>
!vecC = vector<8x1xf32>
!memrefA = memref<1x32x1x2xbf16>
!memrefB = memref<1x1x4x2xbf16>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @brgemm_to_fma_load_bcst_B(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !vecC) -> !vecC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %1 = vector.load %arg0[%c0, %c0, %c0, %c0] : 
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0, %c0, %c0] :
        !memrefB, !vecB
  %3 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %arg2
    : !vecA, !vecB into !vecC
  return %3 : !vecC
}

// CHECK-LABEL: @brgemm_to_fma_load_bcst_B
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.vector_contract_bf16_to_fma
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1x2xbf16>
!vecB = vector<1x1x8x2xbf16>
!vecC = vector<1x1x8xf32>
!memrefA = memref<1x4x1x2xbf16>
!memrefB = memref<1x1x32x2xbf16>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d2)>
func.func @batch_matmul_fma_load(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !vecC) -> !vecC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %1 = vector.load %arg0[%c0, %c0, %c0, %c0] : 
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0, %c0, %c0] :
        !memrefB, !vecB
  %3 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %arg2
    : !vecA, !vecB into !vecC
  return %3 : !vecC
}

// CHECK-LABEL: @batch_matmul_fma_load
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.vector_contract_bf16_to_fma
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x2xbf16>
!vecB = vector<1x8x2xbf16>
!vecC = vector<1x8xf32>
!memrefA = memref<4x1x2xbf16>
!memrefB = memref<1x32x2xbf16>
#map = affine_map<(d4, d1, d2, d3) -> (d1, d3, d4)>
#map1 = affine_map<(d4, d1, d2, d3) -> (d3, d2, d4)>
#map2 = affine_map<(d4, d1, d2, d3) -> (d1, d2)>
func.func @matmul_outer_product_to_fma_load(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !vecC) -> !vecC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %1 = vector.load %arg0[%c0, %c0, %c0] : 
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0, %c0] :
        !memrefB, !vecB
  %3 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %arg2
    : !vecA, !vecB into !vecC
  return %3 : !vecC
}

// CHECK-LABEL: @matmul_outer_product_to_fma_load
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.vector_contract_bf16_to_fma
    } : !transform.any_op
    transform.yield
  }
}
