// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

!vecA = vector<1x1x2xbf16>
!vecB = vector<1x2x16xbf16>
!vecC = vector<1x16xf32>
!memrefA = memref<1x1x2xbf16, strided<[2048, 32, 1], offset: ?>>
!memrefB = memref<1x2x32xbf16, strided<[2048, 64, 1], offset: ?>>
!memrefC = memref<1x32xf32, strided<[64, 1], offset: ?>>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

func.func @brmatmul_acc_mv(%arg0: memref<16x64x32xbf16>, %arg1: memref<16x32x64xbf16>,
                             %arg2: memref<64x64xf32>) -> memref<64x64xf32> {
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %arg3 = %c0 to %c64 step %c1 {
    scf.for %arg4 = %c0 to %c64 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [1, 32] [1, 1]
                        : memref<64x64xf32> to !memrefC
      %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]}
                        : !memrefC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0 {in_bounds = [true, true]}
                        : !memrefC, !vecC

      %4:2 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %2, %arg7 = %3) -> (!vecC, !vecC) {
        %5:2 = scf.for %arg8 = %c0 to %c32 step %c2 iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (!vecC, !vecC) {

          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg8] [1, 1, 2] [1, 1, 1]
                        : memref<16x64x32xbf16> to !memrefA
          %subview_1 = memref.subview %arg1[%arg5, %arg8, %arg4] [1, 2, 32] [1, 1, 1]
                        : memref<16x32x64xbf16> to !memrefB

          %6 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]}
                        : !memrefA, !vecA
          %7 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]}
                        : !memrefB, !vecB
          %8 = vector.transfer_read %subview_1[%c0, %c0, %c16], %1 {in_bounds = [true, true, true]}
                        : !memrefB, !vecB

          %9 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                        ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %6, %7, %arg9
                        {unroll_shape = array<i64: 1, 1, 16, 2>} : !vecA, !vecB into !vecC
          %10 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                        ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %6, %8, %arg10
                        {unroll_shape = array<i64: 1, 1, 16, 2>} : !vecA, !vecB into !vecC

          scf.yield %9, %10 : !vecC, !vecC
        }
        scf.yield %5#0, %5#1 : !vecC, !vecC
      }

      vector.transfer_write %4#1, %subview[%c0, %c16] {in_bounds = [true, true]}
                        : !vecC, !memrefC
      vector.transfer_write %4#0, %subview[%c0, %c0] {in_bounds = [true, true]}
                        : !vecC, !memrefC
    }
  }

  return %arg2 : memref<64x64xf32>
}

// CHECK-LABEL: @brmatmul_acc_mv
// CHECK: arith.constant dense<0.000000e+00> : vector<1x16xf32>
// CHECK: arith.addf
// CHECK-NEXT: vector.transfer_write
// CHECK: arith.addf
// CHECK-NEXT: vector.transfer_write

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.move_accumulator_for_contract_loop
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
func.func @negative_no_loop(
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

// CHECK-LABEL: @negative_no_loop
// CHECK-NOT: arith.constant dense<0.000000e+00> {{.*}}
// CHECK-NOT: arith.addf
// CHECK-NOT: arith.addf

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.move_accumulator_for_contract_loop
    } : !transform.any_op
    transform.yield
  }
}

