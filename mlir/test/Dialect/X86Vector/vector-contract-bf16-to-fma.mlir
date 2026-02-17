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

// CHECK-LABEL: @matmul_to_fma_flat_layout
// CHECK: vector.shuffle{{.*}}[0, 8, 1, 9, 2, 10, 3, 11] : vector<8xf32>, vector<8xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK: memref.subview %arg0[%c0, %c0] {{.*}} : memref<4x1xbf16> to memref<1x1xbf16, {{.*}}>
// CHECK: memref.subview %arg1[%c0, %c0] {{.*}} : memref<1x32xbf16> to memref<1x16xbf16, {{.*}}>
// CHECK: x86vector.avx.bcst_to_f32.packed {{.*}} : memref<1x1xbf16, strided<[1, 1], offset: ?>>
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32 {{.*}} : memref<1x16xbf16, strided<[32, 1], offset: ?>>
// CHECK: vector.fma {{.*}} : vector<8xf32>
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32 {{.*}} : memref<1x16xbf16, strided<[32, 1], offset: ?>>
// CHECK: vector.fma {{.*}} : vector<8xf32>
// CHECK: vector.shuffle{{.*}}[0, 8, 1, 9, 2, 10, 3, 11] : vector<8xf32>, vector<8xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>

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

!vecA = vector<1x1xbf16>
!vecB = vector<1x8xbf16>
!vecC = vector<1x8xf32>
!memrefA = memref<4x1xbf16>
!memrefB = memref<1x32xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @matmul_to_fma_flat_layout_load(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.load %arg0[%c0, %c0] :
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0] :
        !memrefB, !vecB
  %3 = vector.load %arg1[%c0, %c8] :
        !memrefB, !vecB
  %4 = vector.load %arg2[%c0, %c0] :
        !memrefC, !vecC
  %5 = vector.load %arg2[%c0, %c8] :
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

  vector.store %6, %arg2[%c0, %c0] : !memrefC, !vecC
  vector.store %7, %arg2[%c0, %c8] : !memrefC, !vecC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @matmul_to_fma_flat_layout_load
// CHECK: vector.shuffle{{.*}}[0, 8, 1, 9, 2, 10, 3, 11] : vector<8xf32>, vector<8xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK: memref.subview %arg0[%c0, %c0] {{.*}} : memref<4x1xbf16> to memref<1x1xbf16, {{.*}}>
// CHECK: memref.subview %arg1[%c0, %c0] {{.*}} : memref<1x32xbf16> to memref<1x16xbf16, {{.*}}>
// CHECK: x86vector.avx.bcst_to_f32.packed {{.*}} : memref<1x1xbf16, strided<[1, 1], offset: ?>>
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32 {{.*}} : memref<1x16xbf16, strided<[32, 1], offset: ?>>
// CHECK: vector.fma {{.*}} : vector<8xf32>
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32 {{.*}} : memref<1x16xbf16, strided<[32, 1], offset: ?>>
// CHECK: vector.fma {{.*}} : vector<8xf32>
// CHECK: vector.shuffle{{.*}}[0, 8, 1, 9, 2, 10, 3, 11] : vector<8xf32>, vector<8xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>

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

!vecA = vector<1x1x1xbf16>
!vecB = vector<1x1x8xbf16>
!vecC = vector<1x8xf32>
!memrefA = memref<1x1x1xbf16, strided<[2048, 32, 1], offset: ?>>
!memrefB = memref<1x1x16xbf16, strided<[2048, 64, 1], offset: ?>>
!memrefC = memref<1x16xf32, strided<[64, 1], offset: ?>>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

func.func @matmul_to_fma_flat_layout_loop(%arg0: memref<16x64x32xbf16>, %arg1: memref<16x32x64xbf16>, 
              %arg2: memref<64x64xf32>) -> memref<64x64xf32>  {
  %c8 = arith.constant 8 : index
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c64 step %c1 {
    scf.for %arg4 = %c0 to %c64 step %c16 {
      %subview = memref.subview %arg2[%arg3, %arg4] [1, 16] [1, 1] : 
				memref<64x64xf32> to !memrefC
      %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : 
				!memrefC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c8], %0 {in_bounds = [true, true]} : 
				!memrefC, !vecC

      %4:2 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %2, %arg7 = %3) -> (!vecC, !vecC) {
        %5:2 = scf.for %arg8 = %c0 to %c32 step %c1 iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (!vecC, !vecC) {

          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg8] [1, 1, 1] [1, 1, 1] : 
				memref<16x64x32xbf16> to !memrefA
          %subview_1 = memref.subview %arg1[%arg5, %arg8, %arg4] [1, 1, 16] [1, 1, 1] : 
				memref<16x32x64xbf16> to !memrefB

          %6 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1 
				{in_bounds = [true, true, true]} : !memrefA, !vecA
          %7 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1 
				{in_bounds = [true, true, true]} : !memrefB, !vecB
          %8 = vector.transfer_read %subview_1[%c0, %c0, %c8], %1 
				{in_bounds = [true, true, true]} : !memrefB, !vecB

          %9 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = 
				["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} 
				%6, %7, %arg9 {unroll_shape = array<i64: 1, 1, 8, 1>} : !vecA, !vecB into !vecC
          %10 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = 
				["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} 
				%6, %8, %arg10 {unroll_shape = array<i64: 1, 1, 8, 1>} : !vecA, !vecB into !vecC

          scf.yield %9, %10 : !vecC, !vecC
        }
        scf.yield %5#0, %5#1 : !vecC, !vecC
      }

      vector.transfer_write %4#1, %subview[%c0, %c8] {in_bounds = [true, true]} : 
				!vecC, !memrefC
      vector.transfer_write %4#0, %subview[%c0, %c0] {in_bounds = [true, true]} : 
				!vecC, !memrefC
    }
  }

  return %arg2 : memref<64x64xf32>
}

// CHECK-LABEL: @matmul_to_fma_flat_layout_loop
// CHECK: vector.shuffle{{.*}}[0, 8, 1, 9, 2, 10, 3, 11] : vector<8xf32>, vector<8xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma {{.*}} : vector<8xf32>
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma {{.*}} : vector<8xf32>
// CHECK: scf.yield
// CHECK: vector.shuffle{{.*}}[0, 8, 1, 9, 2, 10, 3, 11] : vector<8xf32>, vector<8xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>

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

!vecA = vector<8x1xbf16>
!vecB = vector<1x1xbf16>
!vecC = vector<8x1xf32>
!memrefA = memref<32x1xbf16>
!memrefB = memref<1x4xbf16>
!memrefC = memref<32x4xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @matmul_to_fma_flat_layout_bcstB(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32

  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg0[%c8, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA

  %3 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB

  %4 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %5 = vector.transfer_read %arg2[%c8, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %4
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %2, %3, %5
    : !vecA, !vecB into !vecC

  vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
  vector.transfer_write %7, %arg2[%c8, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @matmul_to_fma_flat_layout_bcstB
// CHECK: vector.shuffle{{.*}}[0, 8, 1, 9, 2, 10, 3, 11] : vector<8xf32>, vector<8xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK: memref.subview %arg1[%c0, %c0] {{.*}} : memref<1x4xbf16> to memref<1x1xbf16, {{.*}}>
// CHECK: memref.subview %arg0[%c0, %c0] {{.*}} : memref<32x1xbf16> to memref<16x1xbf16, {{.*}}>
// CHECK: x86vector.avx.bcst_to_f32.packed {{.*}} : memref<1x1xbf16, strided<[4, 1], offset: ?>>
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32 {{.*}} : memref<16x1xbf16, strided<[1, 1], offset: ?>>
// CHECK: vector.fma {{.*}} : vector<8xf32>
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32 {{.*}} : memref<16x1xbf16, strided<[1, 1], offset: ?>>
// CHECK: vector.fma {{.*}} : vector<8xf32>
// CHECK: vector.shuffle{{.*}}[0, 8, 1, 9, 2, 10, 3, 11] : vector<8xf32>, vector<8xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>

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

!vecA = vector<1x1xbf16>
!vecB = vector<1x8xbf16>
!vecC = vector<1x8xf32>
!memrefA = memref<4x1xbf16>
!memrefB = memref<1x32xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @negative_multiple_vc_users_flat(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.load %arg0[%c0, %c0] :
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0] :
        !memrefB, !vecB
  %3 = vector.load %arg1[%c0, %c8] :
        !memrefB, !vecB
  %4 = vector.load %arg2[%c0, %c0] :
        !memrefC, !vecC
  %5 = vector.load %arg2[%c0, %c8] :
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

  vector.store %6, %arg2[%c0, %c0] : !memrefC, !vecC
  vector.store %7, %arg2[%c0, %c8] : !memrefC, !vecC

  %8 = arith.addf %6, %7 : !vecC
  vector.store %8, %arg2[%c0, %c16] : !memrefC, !vecC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_multiple_vc_users_flat
// CHECK-NOT: vector.shuffle
// CHECK-NOT: vector.fma
// CHECK-NOT: vector.shuffle
// CHECK: vector.contract

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

!vecA = vector<1x1xbf16>
!vecB = vector<1x8xbf16>
!vecC = vector<1x8xf32>
!memrefA = memref<4x1xbf16>
!memrefB = memref<1x32xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @negative_offset_diff_is_not_8(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.load %arg0[%c0, %c0] :
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0] :
        !memrefB, !vecB
  %3 = vector.load %arg1[%c0, %c16] :
        !memrefB, !vecB
  %4 = vector.load %arg2[%c0, %c0] :
        !memrefC, !vecC
  %5 = vector.load %arg2[%c0, %c16] :
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

  vector.store %6, %arg2[%c0, %c0] : !memrefC, !vecC
  vector.store %7, %arg2[%c0, %c16] : !memrefC, !vecC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_offset_diff_is_not_8
// CHECK-NOT: x86vector.avx.bcst_to_f32.packed
// CHECK-NOT: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK-NOT: vector.fma {{.*}} : vector<8xf32>
// CHECK-NOT: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.contract

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

!vecA = vector<1x1xbf16>
!vecB = vector<1x8xbf16>
!vecC = vector<1x8xf32>
!memrefA = memref<4x1xbf16>
!memrefB = memref<1x32xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @negative_vector_contracts_not_in_order(
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

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %5
    : !vecA, !vecB into !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %4
    : !vecA, !vecB into !vecC

  vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
  vector.transfer_write %7, %arg2[%c0, %c8] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_vector_contracts_not_in_order
// CHECK-NOT: x86vector.avx.bcst_to_f32.packed
// CHECK-NOT: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK-NOT: vector.fma {{.*}} : vector<8xf32>
// CHECK-NOT: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.contract

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

!vecA = vector<8x1xbf16>
!vecB = vector<1x1xbf16>
!vecC = vector<8x1xf32>
!memrefA = memref<32x1xbf16>
!memrefB = memref<1x4xbf16>
!memrefC = memref<32x4xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @negative_flat_layout_dynamic_index(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC, %arg3: index) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32

  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg0[%arg3, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA

  %3 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB

  %4 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %5 = vector.transfer_read %arg2[%c8, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %4
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %2, %3, %5
    : !vecA, !vecB into !vecC

  vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
  vector.transfer_write %7, %arg2[%c8, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_flat_layout_dynamic_index
// CHECK-NOT: x86vector.avx.bcst_to_f32.packed
// CHECK-NOT: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK-NOT: vector.fma {{.*}} : vector<8xf32>
// CHECK-NOT: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.contract

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

!vecA = vector<8x2xbf16>
!vecB = vector<2x1xbf16>
!vecC = vector<8x1xf32>
!memrefA = memref<32x2xbf16>
!memrefB = memref<2x4xbf16>
!memrefC = memref<32x4xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @negative_non_unit_K_dim(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32

  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg0[%c8, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA

  %3 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB

  %4 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %5 = vector.transfer_read %arg2[%c8, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %4
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %2, %3, %5
    : !vecA, !vecB into !vecC

  vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
  vector.transfer_write %7, %arg2[%c8, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_non_unit_K_dim
// CHECK-NOT: x86vector.avx.bcst_to_f32.packed
// CHECK-NOT: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK-NOT: vector.fma {{.*}} : vector<8xf32>
// CHECK-NOT: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.contract

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

