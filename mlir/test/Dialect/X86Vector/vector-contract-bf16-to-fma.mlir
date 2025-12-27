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
// CHECK: memref.subview %arg0[%c0, %c0, %c0, 1] {{.*}} : memref<1x4x1x2xbf16> to memref<1x1x1x1xbf16, {{.*}}>
// CHECK: memref.subview %arg0[%c0, %c0, %c0, 0] {{.*}} : memref<1x4x1x2xbf16> to memref<1x1x1x1xbf16, {{.*}}>
// CHECK: memref.subview %arg1[%c0, %c0, %c0, %c0] {{.*}} : memref<1x1x32x2xbf16> to memref<1x1x8x2xbf16, {{.*}}>
// CHECK: x86vector.avx.bcst_to_f32.packed {{.*}} : memref<1x1x1x1xbf16, strided<[8, 2, 2, 1], offset: ?>>
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32 {{.*}} : memref<1x1x8x2xbf16, strided<[64, 64, 2, 1], offset: ?>>
// CHECK: vector.fma {{.*}} : vector<8xf32>
// CHECK: x86vector.avx.bcst_to_f32.packed {{.*}} : memref<1x1x1x1xbf16, strided<[8, 2, 2, 1], offset: ?>>
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32 {{.*}} : memref<1x1x8x2xbf16, strided<[64, 64, 2, 1], offset: ?>>
// CHECK: vector.fma {{.*}} : vector<8xf32>

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

// -----

!vecA = vector<1x1x2xbf16>
!vecB = vector<1x8x2xbf16>
!vecC = vector<1x8xf32>
!memrefA = memref<4x1x2xbf16>
!memrefB = memref<1x32x2xbf16>
#map = affine_map<(d4, d1, d2, d3) -> (d1, d3, d4)>
#map1 = affine_map<(d4, d1, d2, d3) -> (d3, d2, d4)>
#map2 = affine_map<(d4, d1, d2, d3) -> (d1, d2)>
func.func @matmul_dynamic_offset(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !vecC, %arg3: index) -> !vecC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %1 = vector.load %arg0[%arg3, %c0, %c0] :
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

// CHECK-LABEL: @matmul_dynamic_offset
// CHECK: memref.subview %arg0[%arg3, %c0, 1]{{.*}}
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

!vecA = vector<8x1x2xbf16>
!vecB = vector<1x1x2xbf16>
!vecC = vector<8x1xf32>
!memrefA = memref<32x1x2xbf16>
!memrefB = memref<1x4x2xbf16>
#map = affine_map<(d4, d1, d2, d3) -> (d1, d3, d4)>
#map1 = affine_map<(d4, d1, d2, d3) -> (d3, d2, d4)>
#map2 = affine_map<(d4, d1, d2, d3) -> (d1, d2)>
func.func @matmul_to_fma_load_bcst_B(
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

// CHECK-LABEL: @matmul_to_fma_load_bcst_B
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

!vecA = vector<1x1x1x1x2xbf16>
!vecB = vector<1x1x1x8x2xbf16>
!vecC = vector<1x8xf32>
!memrefA = memref<1x1x4x1x2xbf16>
!memrefB = memref<1x1x1x32x2xbf16>
#map = affine_map<(d5, d0, d4, d1, d2, d3) -> (d5, d0, d1, d3, d4)>
#map1 = affine_map<(d5, d0, d4, d1, d2, d3) -> (d5, d0, d3, d2, d4)>
#map2 = affine_map<(d5, d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @many_dimensions(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !vecC) -> !vecC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %1 = vector.load %arg0[%c0, %c0, %c0, %c0, %c0] :
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0, %c0, %c0, %c0] :
        !memrefB, !vecB
  %3 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %arg2
    : !vecA, !vecB into !vecC
  return %3 : !vecC
}

// CHECK-LABEL: @many_dimensions
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
!tensorA = tensor<4x1x2xbf16>
!tensorB = tensor<1x32x2xbf16>
#map = affine_map<(d1, d2, d3, d4) -> (d2, d4, d1)>
#map1 = affine_map<(d1, d2, d3, d4) -> (d4, d3, d1)>
#map2 = affine_map<(d1, d2, d3, d4) -> (d2, d3)>
func.func @negative_tensor_type(%arg0: !tensorA, %arg1: !tensorB, %arg2: !vecC) -> !vecC {
  %0 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %1 = vector.transfer_read %arg0[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} :
        !tensorA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c8, %c0], %0 {in_bounds = [true, true, true]} :
        !tensorB, !vecB
  %3 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %arg2
    : !vecA, !vecB into !vecC
  return %3 : !vecC
}

// CHECK-LABEL: @negative_tensor_type
// CHECK-NOT: vector.fma
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.x86vector.vector_contract_bf16_to_fma
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1x2xbf16>
!vecB = vector<1x1x16x2xbf16>
!vecC = vector<1x1x16xf32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d2)>
func.func @negative_no_memref_src(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_no_memref_src
// CHECK: vector.contract
// CHECK-NOT: vector.fma

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
func.func @negative_non_zero_vnni_offset(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !vecC) -> !vecC
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = ub.poison : bf16
  %1 = vector.load %arg0[%c0, %c0, %c0] :
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0, %c1] :
        !memrefB, !vecB
  %3 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %arg2
    : !vecA, !vecB into !vecC
  return %3 : !vecC
}

// CHECK-LABEL: @negative_non_zero_vnni_offset
// CHECK: vector.contract
// CHECK-NOT: vector.fma

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
#perm0 = affine_map<(d1, d2, d3) -> (d2, d1, d3)>
func.func @negative_perm_map_not_identity(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !vecC) -> !vecC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %1 = vector.transfer_read %arg0[%c0, %c0, %c0], %0 {permutation_map = #perm0,
        in_bounds = [true, true, true]} : !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} :
        !memrefB, !vecB
  %3 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %arg2
    : !vecA, !vecB into !vecC
  return %3 : !vecC
}

// CHECK-LABEL: @negative_perm_map_not_identity
// CHECK: vector.contract
// CHECK-NOT: vector.fma


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
func.func @negative_non_unit_stride(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !vecC) -> !vecC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %subview_1 = memref.subview %arg1[%c0, %c0, %c0] [1, 16, 2] [1, 1, 2] :
               !memrefB to memref<1x16x2xbf16, strided<[64, 2, 2], offset: ?>>

  %1 = vector.transfer_read %arg0[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %subview_1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} :
        memref<1x16x2xbf16, strided<[64, 2, 2], offset: ?>>, !vecB
  %3 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %arg2
    : !vecA, !vecB into !vecC
  return %3 : !vecC
}

// CHECK-LABEL: @negative_non_unit_stride
// CHECK: vector.contract
// CHECK-NOT: vector.fma


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
func.func @negative_out_of_bound(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !vecC, %arg3: index) -> !vecC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16

  %1 = vector.transfer_read %arg0[%c0, %arg3, %c0], %0 {in_bounds = [true, false, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} :
        !memrefB, !vecB
  %3 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %arg2
    : !vecA, !vecB into !vecC
  return %3 : !vecC
}

// CHECK-LABEL: @negative_out_of_bound
// CHECK: vector.contract
// CHECK-NOT: vector.fma


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
func.func @negative_no_dynamic_vnni_offset(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !vecC, %arg3: index) -> !vecC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %1 = vector.load %arg0[%c0, %c0, %arg3] :
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

// CHECK-LABEL: @negative_no_dynamic_vnni_offset
// CHECK: vector.contract
// CHECK-NOT: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.vector_contract_bf16_to_fma
    } : !transform.any_op
    transform.yield
  }
}

