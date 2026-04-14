// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

!vecA = vector<1x16x16x4xi8>
!vecB = vector<1x16x16x4xi8>
!vecC = vector<16x16xi32>
!memrefA = memref<1x32x16x4xi8>
!memrefB = memref<1x16x32x4xi8>
!memrefC = memref<32x32xi32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @brgemm_int8(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : i8
  %32 = ub.poison : i32

  %1 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefB, !vecB

  %3 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} : !memrefC, !vecC

  %4 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %3 : !vecA, !vecB into !vecC

  vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @brgemm_int8
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x64xi8>
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x64xi8>
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x16xi32>
// CHECK: x86.amx.tile_muli
// CHECK: x86.amx.tile_store {{.*}} !x86.amx.tile<16x16xi32>
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<16x16x4xi8>
!vecB = vector<16x16x4xi8>
!vecC = vector<16x16xi32>
!memrefA = memref<32x16x4xi8>
!memrefB = memref<16x32x4xi8>
!memrefC = memref<32x32xi32>
#map = affine_map<(d4, d1, d2, d3) -> (d1, d3, d4)>
#map1 = affine_map<(d4, d1, d2, d3) -> (d3, d2, d4)>
#map2 = affine_map<(d4, d1, d2, d3) -> (d1, d2)>
func.func @matmul_int8(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : i8
  %32 = ub.poison : i32

  %1 = vector.transfer_read %arg0[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} :
        !memrefB, !vecB

  %3 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} : !memrefC, !vecC

  %4 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %3 : !vecA, !vecB into !vecC

  vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @matmul_int8
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x64xi8>
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x64xi8>
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x16xi32>
// CHECK: x86.amx.tile_muli
// CHECK: x86.amx.tile_store {{.*}} !x86.amx.tile<16x16xi32>
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x16x16x2xbf16>
!vecB = vector<1x16x16x2xbf16>
!vecC = vector<16x16xf32>
!memrefA = memref<1x32x16x2xbf16>
!memrefB = memref<1x16x32x2xbf16>
!memrefC = memref<32x32xf32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @brgemm_bf16(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32

  %1 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefB, !vecB

  %3 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} : !memrefC, !vecC

  %4 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %3 : !vecA, !vecB into !vecC

  vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @brgemm_bf16
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x32xbf16>
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x32xbf16>
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x16xf32>
// CHECK: x86.amx.tile_mulf
// CHECK: x86.amx.tile_store {{.*}} !x86.amx.tile<16x16xf32>
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x16x16x2xbf16>
!vecB = vector<1x16x16x2xbf16>
!vecC = vector<1x16x16xf32>
!memrefA = memref<1x32x16x2xbf16>
!memrefB = memref<1x16x32x2xbf16>
!memrefC = memref<1x32x32xf32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d2)>
func.func @batch_matmul_bf16(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32

  %1 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefB, !vecB

  %3 = vector.transfer_read %arg2[%c0, %c0, %c0], %32 {in_bounds = [true, true, true]} : !memrefC, !vecC

  %4 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %3 : !vecA, !vecB into !vecC

  vector.transfer_write %4, %arg2[%c0, %c0, %c0] {in_bounds = [true, true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @batch_matmul_bf16
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x32xbf16>
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x32xbf16>
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x16xf32>
// CHECK: x86.amx.tile_mulf
// CHECK: x86.amx.tile_store {{.*}} !x86.amx.tile<16x16xf32>
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<16x64xi8>
!vecB = vector<64x16xi8>
!vecC = vector<16x16xi32>
!memrefA = memref<32x64xi8>
!memrefB = memref<64x32xi8>
!memrefC = memref<32x32xi32>
#map = affine_map<(d1, d2, d3) -> (d1, d3)>
#map1 = affine_map<(d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d1, d2, d3) -> (d1, d2)>
func.func @online_packing_int8(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : i8
  %32 = ub.poison : i32

  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB

  %3 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} : !memrefC, !vecC

  %4 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %3 : !vecA, !vecB into !vecC

  vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @online_packing_int8
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x64xi8>
// CHECK: scf.for
// CHECK: vector.shuffle{{.*}}[0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55] : vector<32xi8>, vector<32xi8>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63] : vector<32xi8>, vector<32xi8>
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x64xi8>
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x16xi32>
// CHECK: x86.amx.tile_muli
// CHECK: x86.amx.tile_store {{.*}} !x86.amx.tile<16x16xi32>
// CHECK-NOT: vector.contract



module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x16x32xbf16>
!vecB = vector<1x32x16xbf16>
!vecC = vector<16x16xf32>
!memrefA = memref<1x32x32xbf16>
!memrefB = memref<1x32x32xbf16>
!memrefC = memref<32x32xf32>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @online_packing_bf16(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32

  %1 = vector.transfer_read %arg0[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} :
        !memrefB, !vecB

  %3 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} : !memrefC, !vecC

  %4 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %3 : !vecA, !vecB into !vecC

  vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @online_packing_bf16
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x32xbf16>
// CHECK: scf.for
// CHECK: vector.shuffle{{.*}}[0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23] : vector<16xbf16>, vector<16xbf16>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31] : vector<16xbf16>, vector<16xbf16>
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x32xbf16>
// CHECK: x86.amx.tile_load {{.*}} !x86.amx.tile<16x16xf32>
// CHECK: x86.amx.tile_mulf
// CHECK: x86.amx.tile_store {{.*}} !x86.amx.tile<16x16xf32>
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecAB = vector<1x16x16x2xbf16>
!vecC = vector<16x16xf32>
!memrefA = memref<1x32x16x2xbf16, strided<[8192, 128, 2, 1], offset: ?>>
!memrefB = memref<1x16x32x2xbf16, strided<[16384, 256, 2, 1], offset: ?>>
!memrefC = memref<32x32xf32, strided<[128, 1], offset: ?>>

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>

func.func @brgemm_bf16_loop(%arg0: memref<16x64x64x2xbf16>, %arg1: memref<16x64x128x2xbf16>, %arg2: memref<64x128xf32>) -> memref<64x128xf32>  {
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index

  scf.for %arg3 = %c0 to %c64 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {

      %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1] :
                memref<64x128xf32> to !memrefC
      %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} :
                !memrefC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0 {in_bounds = [true, true]} :
                !memrefC, !vecC
      %4 = vector.transfer_read %subview[%c16, %c0], %0 {in_bounds = [true, true]} :
                !memrefC, !vecC
      %5 = vector.transfer_read %subview[%c16, %c16], %0 {in_bounds = [true, true]} :
                !memrefC, !vecC

      %6:4 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5) -> (!vecC, !vecC, !vecC, !vecC) {
        %7:4 = scf.for %arg10 = %c0 to %c64 step %c16 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (!vecC, !vecC, !vecC, !vecC) {

          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg10, 0] [1, 32, 16, 2] [1, 1, 1, 1] :
                memref<16x64x64x2xbf16> to !memrefA
          %subview_1 = memref.subview %arg1[%arg5, %arg10, %arg4, 0] [1, 16, 32, 2] [1, 1, 1, 1] :
                memref<16x64x128x2xbf16> to !memrefB
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} :
                !memrefA, !vecAB
          %9 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} :
                !memrefB, !vecAB

          %10 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %8, %9, %arg11 {unroll_shape = array<i64: 1, 2, 16, 16, 16>} : !vecAB, !vecAB into !vecC

          %11 = vector.transfer_read %subview_1[%c0, %c0, %c16, %c0], %1 {in_bounds = [true, true, true, true]} :
                !memrefB, !vecAB
          %12 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %8, %11, %arg12 {unroll_shape = array<i64: 1, 2, 16, 16, 16>} : !vecAB, !vecAB into !vecC
          %13 = vector.transfer_read %subview_0[%c0, %c16, %c0, %c0], %1 {in_bounds = [true, true, true, true]} :
                !memrefA, !vecAB
          %14 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %13, %9, %arg13 {unroll_shape = array<i64: 1, 2, 16, 16, 16>} : !vecAB, !vecAB into !vecC
          %15 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %13, %11, %arg14 {unroll_shape = array<i64: 1, 2, 16, 16, 16>} : !vecAB, !vecAB into !vecC

          scf.yield %10, %12, %14, %15 : !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %7#0, %7#1, %7#2, %7#3 : !vecC, !vecC, !vecC, !vecC
      }

      vector.transfer_write %6#3, %subview[%c16, %c16] {in_bounds = [true, true]} :
        !vecC, !memrefC
      vector.transfer_write %6#2, %subview[%c16, %c0] {in_bounds = [true, true]} :
        !vecC, !memrefC
      vector.transfer_write %6#1, %subview[%c0, %c16] {in_bounds = [true, true]} :
        !vecC, !memrefC
      vector.transfer_write %6#0, %subview[%c0, %c0] {in_bounds = [true, true]} :
        !vecC, !memrefC
    }
  }

  return %arg2 : memref<64x128xf32>
}

// CHECK-LABEL: @brgemm_bf16_loop
// CHECK-2: scf.for {{.*}} -> (!x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>) { 
// CHECK-4: x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-4: x86.amx.tile_load
// CHECK-4: x86.amx.tile_mulf
// CHECK: scf.yield {{.*}} : !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>
// CHECK-NOT: scf.for {{.*}} vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecAB = vector<16x16x4xi8>
!vecC = vector<16x16xi32>
!memrefA = memref<16x16x4xi8, strided<[256, 4, 1], offset: ?>>
!memrefB = memref<16x32x4xi8, strided<[512, 4, 1], offset: ?>>
!memrefC = memref<16x32xi32, strided<[128, 1], offset: ?>>

#map = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

func.func @matmul_int8_loop(%arg0: memref<64x64x4xi8>, %arg1: memref<64x128x4xi8>, %arg2: memref<64x128xi32>) {
  %0 = ub.poison : i32
  %1 = ub.poison : i8
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  scf.for %arg3 = %c0 to %c64 step %c16 {
    scf.for %arg4 = %c0 to %c128 step %c32 {

      %subview = memref.subview %arg2[%arg3, %arg4] [16, 32] [1, 1] :
                memref<64x128xi32> to !memrefC
      %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} :
                !memrefC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0 {in_bounds = [true, true]} :
                !memrefC, !vecC

      %4:2 = scf.for %arg5 = %c0 to %c64 step %c16 iter_args(%arg6 = %2, %arg7 = %3) -> (!vecC, !vecC) {

        %subview_0 = memref.subview %arg0[%arg3, %arg5, 0] [16, 16, 4] [1, 1, 1] :
                memref<64x64x4xi8> to !memrefA
        %subview_1 = memref.subview %arg1[%arg5, %arg4, 0] [16, 32, 4] [1, 1, 1] :
                memref<64x128x4xi8> to !memrefB
        %5 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} :
                !memrefA, !vecAB
        %6 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} :
                !memrefB, !vecAB
        %7 = vector.transfer_read %subview_1[%c0, %c16, %c0], %1 {in_bounds = [true, true, true]} :
                !memrefB, !vecAB

        %8 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %5, %6, %arg6 {unroll_shape = array<i64: 4, 16, 16, 16>} : !vecAB, !vecAB into !vecC
        %9 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %5, %7, %arg7 {unroll_shape = array<i64: 4, 16, 16, 16>} : !vecAB, !vecAB into !vecC

        scf.yield %8, %9 : !vecC, !vecC
      }

      vector.transfer_write %4#1, %subview[%c0, %c16] {in_bounds = [true, true]} :
                !vecC, !memrefC
      vector.transfer_write %4#0, %subview[%c0, %c0] {in_bounds = [true, true]} :
                !vecC, !memrefC
    }
  }

  return
}

// CHECK-LABEL: @matmul_int8_loop
// CHECK-2: x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
// CHECK: scf.for {{.*}} -> (!x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>) {
// CHECK-3: x86.amx.tile_load
// CHECK-2: x86.amx.tile_muli
// CHECK: scf.yield {{.*}} !x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>
// CHECK-NOT: scf.for {{.*}} vector<16x16xi32>, vector<16x16xi32>
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecAB = vector<1x16x16x4xi8>
!vecC = vector<1x16x16xi32>
!memrefA = memref<1x16x16x4xi8, strided<[16384, 256, 4, 1], offset: ?>>
!memrefB = memref<1x16x32x4xi8, strided<[32768, 512, 4, 1], offset: ?>>
!memrefC = memref<1x16x32xi32, strided<[8192, 128, 1], offset: ?>>

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3)>

func.func @batch_matmul_int8_loop(%arg0: memref<16x64x64x4xi8>, %arg1: memref<16x64x128x4xi8>, %arg2: memref<16x64x128xi32>) {
  %0 = ub.poison : i32
  %1 = ub.poison : i8
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c64 step %c16 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      scf.for %arg5 = %c0 to %c16 step %c1 {

        %subview = memref.subview %arg2[%arg5, %arg3, %arg4] [1, 16, 32] [1, 1, 1] :
                memref<16x64x128xi32> to !memrefC
        %2 = vector.transfer_read %subview[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} :
                !memrefC, !vecC
        %3 = vector.transfer_read %subview[%c0, %c0, %c16], %0 {in_bounds = [true, true, true]} :
                !memrefC, !vecC
        %4:2 = scf.for %arg6 = %c0 to %c64 step %c16 iter_args(%arg7 = %2, %arg8 = %3) -> (!vecC, !vecC) {

          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg6, 0] [1, 16, 16, 4] [1, 1, 1, 1] :
                memref<16x64x64x4xi8> to !memrefA
          %subview_1 = memref.subview %arg1[%arg5, %arg6, %arg4, 0] [1, 16, 32, 4] [1, 1, 1, 1] :
                memref<16x64x128x4xi8> to !memrefB
          %5 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} :
                !memrefA, !vecAB
          %6 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} :
                !memrefB, !vecAB
          %7 = vector.transfer_read %subview_1[%c0, %c0, %c16, %c0], %1 {in_bounds = [true, true, true, true]} :
                !memrefB, !vecAB
          %8 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %5, %6, %arg7 {unroll_shape = array<i64: 1, 4, 16, 16, 16>} : !vecAB, !vecAB into !vecC
          %9 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %5, %7, %arg8 {unroll_shape = array<i64: 1, 4, 16, 16, 16>} : !vecAB, !vecAB into !vecC
          scf.yield %8, %9 : !vecC, !vecC
        }

        vector.transfer_write %4#1, %subview[%c0, %c0, %c16] {in_bounds = [true, true, true]} :
                !vecC, !memrefC
        vector.transfer_write %4#0, %subview[%c0, %c0, %c0] {in_bounds = [true, true, true]} :
                !vecC, !memrefC
      }
    }
  }
  return
}

// CHECK-LABEL: @batch_matmul_int8_loop
// CHECK-2: x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
// CHECK: scf.for {{.*}} -> (!x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>) {
// CHECK-3: x86.amx.tile_load
// CHECK-2: x86.amx.tile_muli
// CHECK: scf.yield {{.*}} !x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>
// CHECK-NOT: scf.for {{.*}} vector<16x16xi32>, vector<16x16xi32>
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x16x32xbf16>
!vecB = vector<1x32x16xbf16>
!vecC = vector<16x16xf32>
!memrefA = memref<1x32x32xbf16, strided<[6144, 96, 1], offset: ?>>
!memrefB = memref<1x32x32xbf16, strided<[12288, 128, 1], offset: ?>>
!memrefC = memref<32x32xf32, strided<[128, 1], offset: ?>>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

func.func @online_packing_bf16_loop(%arg0: memref<16x64x96xbf16>, %arg1: memref<16x96x128xbf16>, %arg2: memref<64x128xf32>) -> memref<64x128xf32> {
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %c96 = arith.constant 96 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c64 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {

      %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1] :
                memref<64x128xf32> to !memrefC
      %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} :
                !memrefC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0 {in_bounds = [true, true]} :
                !memrefC, !vecC
      %4 = vector.transfer_read %subview[%c16, %c0], %0 {in_bounds = [true, true]} :
                !memrefC, !vecC
      %5 = vector.transfer_read %subview[%c16, %c16], %0 {in_bounds = [true, true]} :
                !memrefC, !vecC

      %6:4 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5) -> (!vecC, !vecC, !vecC, !vecC) {
        %7:4 = scf.for %arg10 = %c0 to %c96 step %c32 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (!vecC, !vecC, !vecC, !vecC) {

          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg10] [1, 32, 32] [1, 1, 1] :
                memref<16x64x96xbf16> to !memrefA
          %subview_1 = memref.subview %arg1[%arg5, %arg10, %arg4] [1, 32, 32] [1, 1, 1] :
                memref<16x96x128xbf16> to !memrefB
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} :
                !memrefA, !vecA
          %9 = vector.transfer_read %subview_0[%c0, %c16, %c0], %1 {in_bounds = [true, true, true]} :
                !memrefA, !vecA
          %10 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} :
                !memrefB, !vecB
          %11 = vector.transfer_read %subview_1[%c0, %c0, %c16], %1 {in_bounds = [true, true, true]} :
                !memrefB, !vecB

          %12 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %8, %10, %arg11 {unroll_shape = array<i64: 1, 16, 16, 32>} : !vecA, !vecB into !vecC
          %13 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %8, %11, %arg12 {unroll_shape = array<i64: 1, 16, 16, 32>} : !vecA, !vecB into !vecC
          %14 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %9, %10, %arg13 {unroll_shape = array<i64: 1, 16, 16, 32>} : !vecA, !vecB into !vecC
          %15 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %9, %11, %arg14 {unroll_shape = array<i64: 1, 16, 16, 32>} : !vecA, !vecB into !vecC

          scf.yield %12, %13, %14, %15 : !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %7#0, %7#1, %7#2, %7#3 : !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %6#3, %subview[%c16, %c16] {in_bounds = [true, true]} :
                !vecC, !memrefC
      vector.transfer_write %6#2, %subview[%c16, %c0] {in_bounds = [true, true]} :
                !vecC, !memrefC
      vector.transfer_write %6#1, %subview[%c0, %c16] {in_bounds = [true, true]} :
                !vecC, !memrefC
      vector.transfer_write %6#0, %subview[%c0, %c0] {in_bounds = [true, true]} :
                !vecC, !memrefC
    }
  }
  %alloc = memref.alloc() : memref<64x128xf32>
  memref.copy %arg2, %alloc : memref<64x128xf32> to memref<64x128xf32>
  return %alloc : memref<64x128xf32>
}

// CHECK-LABEL: @online_packing_bf16_loop
// CHECK-COUNT-4: x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-COUNT-4: scf.for {{.*}} -> (!x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>) {
// CHECK: vector.shuffle{{.*}}[0, 32, 1, 33, 2, 34, 3, 35, 8, 40, 9, 41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 36, 5, 37, 6, 38, 7, 39, 12, 44, 13, 45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: x86.amx.tile_load
// CHECK: x86.amx.tile_mulf
// CHECK: scf.yield {{.*}} : !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK-NOT: scf.for {{.*}} vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>
// CHECK-NOT: vector.contract



module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<16x64xi8>
!vecB = vector<64x16xi8>
!vecC = vector<16x16xi32>
!memrefA = memref<32x64xi8, strided<[256, 1], offset: ?>>
!memrefB = memref<64x32xi8, strided<[128, 1], offset: ?>>
!memrefC = memref<32x32xi32, strided<[128, 1], offset: ?>>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @online_packing_int8_matmul_loop(%arg0: memref<64x256xi8>, %arg1: memref<256x128xi8>, %arg2: memref<64x128xi32>) -> memref<64x128xi32> {
  %c16 = arith.constant 16 : index
  %0 = ub.poison : i32
  %1 = ub.poison : i8
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c32 = arith.constant 32 : index
  scf.for %arg3 = %c0 to %c64 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1] : memref<64x128xi32> to !memrefC
      %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : !memrefC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0 {in_bounds = [true, true]} : !memrefC, !vecC
      %4 = vector.transfer_read %subview[%c16, %c0], %0 {in_bounds = [true, true]} : !memrefC, !vecC
      %5 = vector.transfer_read %subview[%c16, %c16], %0 {in_bounds = [true, true]} : !memrefC, !vecC
      %6:4 = scf.for %arg5 = %c0 to %c256 step %c64 iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5) -> (!vecC, !vecC, !vecC, !vecC) {
        %subview_0 = memref.subview %arg0[%arg3, %arg5] [32, 64] [1, 1] : memref<64x256xi8> to !memrefA
        %subview_1 = memref.subview %arg1[%arg5, %arg4] [64, 32] [1, 1] : memref<256x128xi8> to !memrefB
        %7 = vector.transfer_read %subview_0[%c0, %c0], %1 {in_bounds = [true, true]} : !memrefA, !vecA
        %8 = vector.transfer_read %subview_0[%c16, %c0], %1 {in_bounds = [true, true]} : !memrefA, !vecA
        %9 = vector.transfer_read %subview_1[%c0, %c0], %1 {in_bounds = [true, true]} : !memrefB, !vecB
        %10 = vector.transfer_read %subview_1[%c0, %c16], %1 {in_bounds = [true, true]} : !memrefB, !vecB
        %11 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %7, %9, %arg6 {unroll_shape = array<i64: 16, 16, 64>} : !vecA, !vecB into !vecC
        %12 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %7, %10, %arg7 {unroll_shape = array<i64: 16, 16, 64>} : !vecA, !vecB into !vecC
        %13 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %8, %9, %arg8 {unroll_shape = array<i64: 16, 16, 64>} : !vecA, !vecB into !vecC
        %14 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %8, %10, %arg9 {unroll_shape = array<i64: 16, 16, 64>} : !vecA, !vecB into !vecC
        scf.yield %11, %12, %13, %14 : !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %6#3, %subview[%c16, %c16] {in_bounds = [true, true]} : !vecC, !memrefC
      vector.transfer_write %6#2, %subview[%c16, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
      vector.transfer_write %6#1, %subview[%c0, %c16] {in_bounds = [true, true]} : !vecC, !memrefC
      vector.transfer_write %6#0, %subview[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
    }
  }
  %alloc = memref.alloc() : memref<64x128xi32>
  memref.copy %arg2, %alloc : memref<64x128xi32> to memref<64x128xi32>
  return %alloc : memref<64x128xi32>
}

// CHECK-LABEL: @online_packing_int8_matmul_loop
// CHECK-COUNT-4: x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
// CHECK: scf.for {{.*}} -> (!x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>) {
// CHECK: vector.shuffle{{.*}}[0, 32, 64, 96, 1, 33, 65, 97, 2, 34, 66, 98, 3, 35, 67, 99, 8, 40, 72, 104, 9, 41, 73, 105, 10, 42, 74, 106, 11, 43, 75, 107, 16, 48, 80, 112, 17, 49, 81, 113, 18, 50, 82, 114, 19, 51, 83, 115, 24, 56, 88, 120, 25, 57, 89, 121, 26, 58, 90, 122, 27, 59, 91, 123] : vector<64xi8>, vector<64xi8>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 36, 68, 100, 5, 37, 69, 101, 6, 38, 70, 102, 7, 39, 71, 103, 12, 44, 76, 108, 13, 45, 77, 109, 14, 46, 78, 110, 15, 47, 79, 111, 20, 52, 84, 116, 21, 53, 85, 117, 22, 54, 86, 118, 23, 55, 87, 119, 28, 60, 92, 124, 29, 61, 93, 125, 30, 62, 94, 126, 31, 63, 95, 127] : vector<64xi8>, vector<64xi8>
// CHECK: x86.amx.tile_load
// CHECK: x86.amx.tile_muli
// CHECK: scf.yield {{.*}} !x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xi32>, vector<16xi32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xi32>, vector<16xi32>
// CHECK-NOT: scf.for {{.*}} vector<16x16xi32>, vector<16x16xi32>, vector<16x16xi32>, vector<16x16xi32>
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x16x16x4xi8>
!vecB = vector<1x16x16x4xi8>
!vecC = vector<16x16xi32>
!memrefA = memref<1x32x16x4xi8>
!memrefB = memref<1x16x32x4xi8>
!memrefC = memref<32x32xi32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @negative_invalid_vc_kind(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : i8
  %32 = ub.poison : i32

  %1 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefB, !vecB

  %3 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} : !memrefC, !vecC

  %4 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<mul>}
    %1, %2, %3 : !vecA, !vecB into !vecC

  vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_invalid_vc_kind
// CHECK-NOT: x86.amx.tile_load {{.*}} !x86.amx.tile<16x64xi8>
// CHECK-NOT: x86.amx.tile_muli
// CHECK-NOT: x86.amx.tile_store {{.*}} !x86.amx.tile<16x16xi32>
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x32x16x2xbf16>
!vecB = vector<1x16x32x2xbf16>
!vecC = vector<1x32x32xf32>
!memrefA = memref<1x32x16x2xbf16>
!memrefB = memref<1x16x32x2xbf16>
!memrefC = memref<1x32x32xf32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d2)>
func.func @negative_wrong_dimensions(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32

  %1 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefB, !vecB

  %3 = vector.transfer_read %arg2[%c0, %c0, %c0], %32 {in_bounds = [true, true, true]} : !memrefC, !vecC

  %4 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %3 : !vecA, !vecB into !vecC

  vector.transfer_write %4, %arg2[%c0, %c0, %c0] {in_bounds = [true, true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_wrong_dimensions
// CHECK-NOT: x86.amx.tile_load
// CHECK-NOT: x86.amx.tile_mulf
// CHECK-NOT: x86.amx.tile_store 
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<16x16x4xi8>
!vecB = vector<16x16x4xi8>
!vecC = vector<16x16xi32>
!memrefB = memref<16x32x4xi8>
!memrefC = memref<32x32xi32>
#map = affine_map<(d4, d1, d2, d3) -> (d1, d3, d4)>
#map1 = affine_map<(d4, d1, d2, d3) -> (d3, d2, d4)>
#map2 = affine_map<(d4, d1, d2, d3) -> (d1, d2)>
func.func @negative_no_memref_source_LHS(
  %arg0: !vecA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : i8
  %32 = ub.poison : i32

  %2 = vector.transfer_read %arg1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} :
        !memrefB, !vecB

  %3 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} : !memrefC, !vecC

  %4 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %2, %3 : !vecA, !vecB into !vecC

  vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_no_memref_source_LHS
// CHECK-NOT: x86.amx.tile_load {{.*}} !x86.amx.tile<16x64xi8>
// CHECK-NOT: x86.amx.tile_muli
// CHECK-NOT: x86.amx.tile_store {{.*}} !x86.amx.tile<16x16xi32>
// CHECK: vector.contract


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x16x32xbf16>
!vecB = vector<1x32x32xbf16>
!vecC = vector<16x32xf32>
!memrefA = memref<1x32x32xbf16>
!memrefB = memref<1x32x32xbf16>
!memrefC = memref<32x32xf32>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @negative_wrong_dimensions_online_packing(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32

  %1 = vector.transfer_read %arg0[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} :
        !memrefB, !vecB

  %3 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} : !memrefC, !vecC

  %4 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %3 : !vecA, !vecB into !vecC

  vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_wrong_dimensions_online_packing
// CHECK-NOT: x86.amx.tile_load
// CHECK-NOT: vector.shuffle
// CHECK-NOT: x86.amx.tile_mulf
// CHECK-NOT: x86.amx.tile_store
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecAB = vector<16x16x4xi8>
!vecC = vector<16x16xi32>
!memrefA = memref<16x16x4xi8, strided<[256, 4, 1], offset: ?>>
!memrefB = memref<16x32x4xi8, strided<[512, 4, 1], offset: ?>>
!memrefC = memref<16x32xi32, strided<[128, 1], offset: ?>>

#map = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

func.func @negative_VCs_LHS_src_differ(%arg0: memref<64x64x4xi8>, %arg1: memref<64x128x4xi8>, %arg2: memref<64x128xi32>, %arg13: memref<64x64x4xi8>) {
  %0 = ub.poison : i32
  %1 = ub.poison : i8
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  scf.for %arg3 = %c0 to %c64 step %c16 {
    scf.for %arg4 = %c0 to %c128 step %c32 {

      %subview = memref.subview %arg2[%arg3, %arg4] [16, 32] [1, 1] :
                memref<64x128xi32> to !memrefC
      %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} :
                !memrefC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0 {in_bounds = [true, true]} :
                !memrefC, !vecC

      %4:2 = scf.for %arg5 = %c0 to %c64 step %c16 iter_args(%arg6 = %2, %arg7 = %3) -> (!vecC, !vecC) {

        %subview_0 = memref.subview %arg0[%arg3, %arg5, 0] [16, 16, 4] [1, 1, 1] :
                memref<64x64x4xi8> to !memrefA
        %subview_negative = memref.subview %arg13[%arg3, %arg5, 0] [16, 16, 4] [1, 1, 1] :
                memref<64x64x4xi8> to !memrefA
        %subview_1 = memref.subview %arg1[%arg5, %arg4, 0] [16, 32, 4] [1, 1, 1] :
                memref<64x128x4xi8> to !memrefB
        %5 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} :
                !memrefA, !vecAB
        %odd_load = vector.transfer_read %subview_negative[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} :
                !memrefA, !vecAB
        %6 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} :
                !memrefB, !vecAB
        %7 = vector.transfer_read %subview_1[%c0, %c16, %c0], %1 {in_bounds = [true, true, true]} :
                !memrefB, !vecAB

        %8 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %5, %6, %arg6 {unroll_shape = array<i64: 4, 16, 16, 16>} : !vecAB, !vecAB into !vecC
        %9 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %odd_load, %7, %arg7 {unroll_shape = array<i64: 4, 16, 16, 16>} : !vecAB, !vecAB into !vecC

        scf.yield %8, %9 : !vecC, !vecC
      }

      vector.transfer_write %4#1, %subview[%c0, %c16] {in_bounds = [true, true]} :
                !vecC, !memrefC
      vector.transfer_write %4#0, %subview[%c0, %c0] {in_bounds = [true, true]} :
                !vecC, !memrefC
    }
  }

  return
}

// CHECK-LABEL: @negative_VCs_LHS_src_differ
// CHECK-NOT: x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
// CHECK-NOT: scf.for {{.*}} -> (!x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>) {
// CHECK-NOT: x86.amx.tile_load
// CHECK-NOT: x86.amx.tile_muli
// CHECK-NOT: scf.yield {{.*}} !x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>
// CHECK: scf.for {{.*}} -> (vector<16x16xi32>, vector<16x16xi32>) {
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x16x16x4xi8>
!vecB = vector<1x16x32x4xi8>
!vecC = vector<1x16x32xi32>
!memrefA = memref<1x16x16x4xi8, strided<[16384, 256, 4, 1], offset: ?>>
!memrefB = memref<1x16x32x4xi8, strided<[32768, 512, 4, 1], offset: ?>>
!memrefC = memref<1x16x32xi32, strided<[8192, 128, 1], offset: ?>>

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3)>

func.func @negative_wrong_N_dim(%arg0: memref<16x64x64x4xi8>, %arg1: memref<16x64x128x4xi8>, %arg2: memref<16x64x128xi32>) {
  %0 = ub.poison : i32
  %1 = ub.poison : i8
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c64 step %c16 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      scf.for %arg5 = %c0 to %c16 step %c1 {

        %subview = memref.subview %arg2[%arg5, %arg3, %arg4] [1, 16, 32] [1, 1, 1] :
                memref<16x64x128xi32> to !memrefC
        %2 = vector.transfer_read %subview[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} :
                !memrefC, !vecC
        %3 = scf.for %arg6 = %c0 to %c64 step %c16 iter_args(%arg7 = %2) -> (!vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg6, 0] [1, 16, 16, 4] [1, 1, 1, 1] :
                memref<16x64x64x4xi8> to !memrefA
          %subview_1 = memref.subview %arg1[%arg5, %arg6, %arg4, 0] [1, 16, 32, 4] [1, 1, 1, 1] :
                memref<16x64x128x4xi8> to !memrefB
          %4 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} :
                !memrefA, !vecA
          %5 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} :
                !memrefB, !vecB
          %6 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %4, %5, %arg7 {unroll_shape = array<i64: 1, 4, 16, 32, 16>} : !vecA, !vecB into !vecC
          scf.yield %6 : !vecC
        }
        vector.transfer_write %3, %subview[%c0, %c0, %c0] {in_bounds = [true, true, true]} :
                !vecC, !memrefC
      }
    }
  }
  return
}

// CHECK-LABEL: @negative_wrong_N_dim
// CHECK-NOT: x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
// CHECK-NOT: x86.amx.tile_load
// CHECK-NOT: x86.amx.tile_muli
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecAB = vector<1x1x16x16x4xi8>
!vecC = vector<16x16xi32>
!memrefA = memref<1x1x16x16x4xi8, strided<[262144, 16384, 256, 4, 1], offset: ?>>
!memrefB = memref<1x1x16x32x4xi8, strided<[524288, 32768, 512, 4, 1], offset: ?>>
!memrefC = memref<16x32xi32, strided<[128, 1], offset: ?>>

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4)>

func.func @negative_reduction_loop_depth_3(%arg0: memref<2x16x64x64x4xi8>, %arg1: memref<2x16x64x128x4xi8>, %arg2: memref<64x128xi32>) attributes {dlti.target_system_spec = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"reg_gemm_unroll" = [16, 16, 16]>>} {
  %0 = ub.poison : i32
  %1 = ub.poison : i8
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c64 step %c16 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [16, 32] [1, 1] :
                memref<64x128xi32> to !memrefC
      %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} :
                !memrefC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0 {in_bounds = [true, true]} :
                !memrefC, !vecC

      %4:2 = scf.for %arg5 = %c0 to %c2 step %c1 iter_args(%arg6 = %2, %arg7 = %3) -> (!vecC, !vecC) {
        %5:2 = scf.for %arg8 = %c0 to %c16 step %c1 iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (!vecC, !vecC) {
          %6:2 = scf.for %arg11 = %c0 to %c64 step %c16 iter_args(%arg12 = %arg9, %arg13 = %arg10) -> (!vecC, !vecC) {
            %subview_0 = memref.subview %arg0[%arg5, %arg8, %arg3, %arg11, 0] [1, 1, 16, 16, 4] [1, 1, 1, 1, 1] :
                memref<2x16x64x64x4xi8> to !memrefA
            %subview_1 = memref.subview %arg1[%arg5, %arg8, %arg11, %arg4, 0] [1, 1, 16, 32, 4] [1, 1, 1, 1, 1] :
                memref<2x16x64x128x4xi8> to !memrefB
            %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true, true]} :
                !memrefA, !vecAB
            %8 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true, true]} :
                !memrefB, !vecAB
            %9 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c16, %c0], %1 {in_bounds = [true, true, true, true, true]} :
                !memrefB, !vecAB

            %10 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %7, %8, %arg12 {unroll_shape = array<i64: 1, 1, 4, 16, 16, 16>} : !vecAB, !vecAB into !vecC
            %11 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %7, %9, %arg13 {unroll_shape = array<i64: 1, 1, 4, 16, 16, 16>} : !vecAB, !vecAB into !vecC
            scf.yield %10, %11 : !vecC, !vecC
          }
          scf.yield %6#0, %6#1 : !vecC, !vecC
        }
        scf.yield %5#0, %5#1 : !vecC, !vecC
      }

      vector.transfer_write %4#1, %subview[%c0, %c16] {in_bounds = [true, true]} :
                !vecC, !memrefC
      vector.transfer_write %4#0, %subview[%c0, %c0] {in_bounds = [true, true]} :
                !vecC, !memrefC
    }
  }
  return
}


// CHECK-LABEL: @negative_reduction_loop_depth_3
// CHECK-NOT: x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
// CHECK-NOT: x86.amx.tile_load
// CHECK-NOT: x86.amx.tile_muli
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x16x16x2xf16>
!vecB = vector<1x16x16x2xf16>
!vecC = vector<16x16xf32>
!memrefA = memref<1x32x16x2xf16>
!memrefB = memref<1x16x32x2xf16>
!memrefC = memref<32x32xf32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @negative_wrong_type_f16(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : f16
  %32 = ub.poison : f32

  %1 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefB, !vecB

  %3 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} : !memrefC, !vecC

  %4 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %3 : !vecA, !vecB into !vecC

  vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_wrong_type_f16
// CHECK-NOT: x86.amx.tile_load
// CHECK-NOT: x86.amx.tile_mulf
// CHECK-NOT: x86.amx.tile_store
// CHECK: vector.contract


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x16x16x2xbf16>
!vecB = vector<1x16x16x2xbf16>
!vecC = vector<16x16xbf16>
!memrefA = memref<1x32x16x2xbf16>
!memrefB = memref<1x16x32x2xbf16>
!memrefC = memref<32x32xbf16>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @negative_wrong_acc_type_bf16(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : bf16

  %1 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} :
        !memrefB, !vecB

  %3 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} : !memrefC, !vecC

  %4 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %3 : !vecA, !vecB into !vecC

  vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_wrong_acc_type_bf16
// CHECK-NOT: x86.amx.tile_load
// CHECK-NOT: x86.amx.tile_mulf
// CHECK-NOT: x86.amx.tile_store
// CHECK: vector.contract


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<16x64xi8>
!vecB = vector<64x16xi8>
!vecC = vector<16x16xi32>
!memrefA = memref<16x64xi8, strided<[256, 1], offset: ?>>
!memrefB = memref<64x32xi8, strided<[128, 1], offset: ?>>
!memrefC = memref<16x32xi32, strided<[128, 1], offset: ?>>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @negative_vc_wrong_order_no_pair(%arg0: memref<64x256xi8>, %arg1: memref<256x128xi8>, %arg2: memref<64x128xi32>) -> memref<64x128xi32> {
  %0 = ub.poison : i32
  %1 = ub.poison : i8
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  scf.for %arg3 = %c0 to %c64 step %c16 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [16, 32] [1, 1]
                : memref<64x128xi32> to !memrefC
      %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]}
                : !memrefC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0 {in_bounds = [true, true]}
                : !memrefC, !vecC
      %4:2 = scf.for %arg5 = %c0 to %c256 step %c64 iter_args(%arg6 = %2, %arg7 = %3) -> (!vecC, !vecC) {
        %subview_0 = memref.subview %arg0[%arg3, %arg5] [16, 64] [1, 1]
                : memref<64x256xi8> to !memrefA
        %subview_1 = memref.subview %arg1[%arg5, %arg4] [64, 32] [1, 1]
                : memref<256x128xi8> to !memrefB
        %5 = vector.transfer_read %subview_0[%c0, %c0], %1 {in_bounds = [true, true]}
                : !memrefA, !vecA
        %6 = vector.transfer_read %subview_1[%c0, %c0], %1 {in_bounds = [true, true]}
                : !memrefB, !vecB
        %7 = vector.transfer_read %subview_1[%c0, %c16], %1 {in_bounds = [true, true]}
                : !memrefB, !vecB
        %8 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %5, %7, %arg7 {unroll_shape = array<i64: 16, 16, 64>} : !vecA, !vecB into !vecC
        %9 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %5, %6, %arg6 {unroll_shape = array<i64: 16, 16, 64>} : !vecA, !vecB into !vecC
        scf.yield %9, %8 : !vecC, !vecC
      }
      vector.transfer_write %4#1, %subview[%c0, %c16] {in_bounds = [true, true]}
                : !vecC, !memrefC
      vector.transfer_write %4#0, %subview[%c0, %c0] {in_bounds = [true, true]}
                : !vecC, !memrefC
    }
  }
  %alloc = memref.alloc() : memref<64x128xi32>
  memref.copy %arg2, %alloc : memref<64x128xi32> to memref<64x128xi32>
  return %alloc : memref<64x128xi32>
}


// CHECK-LABEL: @negative_vc_wrong_order_no_pair
// CHECK-NOT: x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
// CHECK-NOT: x86.amx.tile_load
// CHECK-NOT: x86.amx.tile_muli
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<16x64xi8>
!vecB = vector<64x16xi8>
!vecC = vector<16x16xi32>
!memrefA = memref<32x64xi8, strided<[256, 1], offset: ?>>
!memrefB = memref<64x16xi8, strided<[128, 1], offset: ?>>
!memrefC = memref<32x16xi32, strided<[128, 1], offset: ?>>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @negative_vc_no_pair(%arg0: memref<64x256xi8>, %arg1: memref<256x128xi8>, %arg2: memref<64x128xi32>) -> memref<64x128xi32> {
  %0 = ub.poison : i32
  %1 = ub.poison : i8
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  scf.for %arg3 = %c0 to %c64 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c16 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 16] [1, 1] : memref<64x128xi32> to !memrefC
      %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]}
                : !memrefC, !vecC
      %3 = vector.transfer_read %subview[%c16, %c0], %0 {in_bounds = [true, true]}
                : !memrefC, !vecC
      %4:2 = scf.for %arg5 = %c0 to %c256 step %c64 iter_args(%arg6 = %2, %arg7 = %3) -> (!vecC, !vecC) {
        %subview_0 = memref.subview %arg0[%arg3, %arg5] [32, 64] [1, 1]
                : memref<64x256xi8> to !memrefA
        %subview_1 = memref.subview %arg1[%arg5, %arg4] [64, 16] [1, 1]
                : memref<256x128xi8> to !memrefB
        %5 = vector.transfer_read %subview_0[%c0, %c0], %1 {in_bounds = [true, true]}
                : !memrefA, !vecA
        %6 = vector.transfer_read %subview_0[%c16, %c0], %1 {in_bounds = [true, true]}
                : !memrefA, !vecA
        %7 = vector.transfer_read %subview_1[%c0, %c0], %1 {in_bounds = [true, true]}
                : !memrefB, !vecB
        %8 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %5, %7, %arg6 {unroll_shape = array<i64: 16, 16, 64>} : !vecA, !vecB into !vecC
        %9 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %6, %7, %arg7 {unroll_shape = array<i64: 16, 16, 64>} : !vecA, !vecB into !vecC
        scf.yield %8, %9 : !vecC, !vecC
      }
      vector.transfer_write %4#1, %subview[%c16, %c0] {in_bounds = [true, true]}
                : !vecC, !memrefC
      vector.transfer_write %4#0, %subview[%c0, %c0] {in_bounds = [true, true]}
                : !vecC, !memrefC
    }
  }
  %alloc = memref.alloc() : memref<64x128xi32>
  memref.copy %arg2, %alloc : memref<64x128xi32> to memref<64x128xi32>
  return %alloc : memref<64x128xi32>
}

// CHECK-LABEL: @negative_vc_no_pair
// CHECK-NOT: x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
// CHECK-NOT: x86.amx.tile_load
// CHECK-NOT: x86.amx.tile_muli
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}
