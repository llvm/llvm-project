// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s


#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>

  module {
    func.func @brgemm_amx(%arg0: memref<16x64x64x2xbf16>, %arg1: memref<16x64x128x2xbf16>, %arg2: memref<64x128xf32>) -> memref<64x128xf32> attributes {dlti.target_system_spec = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"reg_gemm_unroll" = [16, 16, 16]>>} {
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
          %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1] : memref<64x128xf32> to memref<32x32xf32, strided<[128, 1], offset: ?>>
          %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : memref<32x32xf32, strided<[128, 1], offset: ?>>, vector<16x16xf32>
          %3 = vector.transfer_read %subview[%c0, %c16], %0 {in_bounds = [true, true]} : memref<32x32xf32, strided<[128, 1], offset: ?>>, vector<16x16xf32>
          %4 = vector.transfer_read %subview[%c16, %c0], %0 {in_bounds = [true, true]} : memref<32x32xf32, strided<[128, 1], offset: ?>>, vector<16x16xf32>
          %5 = vector.transfer_read %subview[%c16, %c16], %0 {in_bounds = [true, true]} : memref<32x32xf32, strided<[128, 1], offset: ?>>, vector<16x16xf32>
          %6:4 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5) -> (vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>) {
            %7:4 = scf.for %arg10 = %c0 to %c64 step %c16 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>) {
              %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg10, 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<16x64x64x2xbf16> to memref<1x32x16x2xbf16, strided<[8192, 128, 2, 1], offset: ?>>
              %subview_1 = memref.subview %arg1[%arg5, %arg10, %arg4, 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<16x64x128x2xbf16> to memref<1x16x32x2xbf16, strided<[16384, 256, 2, 1], offset: ?>>
              %8 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} : memref<1x32x16x2xbf16, strided<[8192, 128, 2, 1], offset: ?>>, vector<1x16x16x2xbf16>
              %9 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} : memref<1x16x32x2xbf16, strided<[16384, 256, 2, 1], offset: ?>>, vector<1x16x16x2xbf16>
              %10 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %8, %9, %arg11 {unroll_shape = array<i64: 1, 2, 16, 16, 16>} : vector<1x16x16x2xbf16>, vector<1x16x16x2xbf16> into vector<16x16xf32>
              %11 = vector.transfer_read %subview_1[%c0, %c0, %c16, %c0], %1 {in_bounds = [true, true, true, true]} : memref<1x16x32x2xbf16, strided<[16384, 256, 2, 1], offset: ?>>, vector<1x16x16x2xbf16>
              %12 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %8, %11, %arg12 {unroll_shape = array<i64: 1, 2, 16, 16, 16>} : vector<1x16x16x2xbf16>, vector<1x16x16x2xbf16> into vector<16x16xf32>
              %13 = vector.transfer_read %subview_0[%c0, %c16, %c0, %c0], %1 {in_bounds = [true, true, true, true]} : memref<1x32x16x2xbf16, strided<[8192, 128, 2, 1], offset: ?>>, vector<1x16x16x2xbf16>
              %14 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %13, %9, %arg13 {unroll_shape = array<i64: 1, 2, 16, 16, 16>} : vector<1x16x16x2xbf16>, vector<1x16x16x2xbf16> into vector<16x16xf32>
              %15 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %13, %11, %arg14 {unroll_shape = array<i64: 1, 2, 16, 16, 16>} : vector<1x16x16x2xbf16>, vector<1x16x16x2xbf16> into vector<16x16xf32>
              scf.yield %10, %12, %14, %15 : vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>
            }
            scf.yield %7#0, %7#1, %7#2, %7#3 : vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>
          }
          vector.transfer_write %6#3, %subview[%c16, %c16] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[128, 1], offset: ?>>
          vector.transfer_write %6#2, %subview[%c16, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[128, 1], offset: ?>>
          vector.transfer_write %6#1, %subview[%c0, %c16] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[128, 1], offset: ?>>
          vector.transfer_write %6#0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[128, 1], offset: ?>>
        }
      }
      %alloc = memref.alloc() : memref<64x128xf32>
      memref.copy %arg2, %alloc : memref<64x128xf32> to memref<64x128xf32>
      return %alloc : memref<64x128xf32>
    }
  }

// CHECK-LABEL: @brgemm_amx
// CHECK: amx.tile_mulf
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.amx.vector_contract_to_packed_type_tiled_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

  module {
    func.func @batch_amx(%arg0: memref<64x64x2xbf16>, %arg1: memref<64x128x2xbf16>, %arg2: memref<64x128xf32>) -> memref<64x128xf32> attributes {dlti.target_system_spec = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"reg_gemm_unroll" = [16, 16, 16]>>} {
      %0 = ub.poison : f32
      %1 = ub.poison : bf16
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c16 = arith.constant 16 : index
      scf.for %arg3 = %c0 to %c64 step %c32 {
        scf.for %arg4 = %c0 to %c128 step %c32 {
          %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1] : memref<64x128xf32> to memref<32x32xf32, strided<[128, 1], offset: ?>>
          %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : memref<32x32xf32, strided<[128, 1], offset: ?>>, vector<16x16xf32>
          %3 = vector.transfer_read %subview[%c0, %c16], %0 {in_bounds = [true, true]} : memref<32x32xf32, strided<[128, 1], offset: ?>>, vector<16x16xf32>
          %4 = vector.transfer_read %subview[%c16, %c0], %0 {in_bounds = [true, true]} : memref<32x32xf32, strided<[128, 1], offset: ?>>, vector<16x16xf32>
          %5 = vector.transfer_read %subview[%c16, %c16], %0 {in_bounds = [true, true]} : memref<32x32xf32, strided<[128, 1], offset: ?>>, vector<16x16xf32>
          %6:4 = scf.for %arg5 = %c0 to %c64 step %c16 iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5) -> (vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>) {
            %subview_0 = memref.subview %arg0[%arg3, %arg5, 0] [32, 16, 2] [1, 1, 1] : memref<64x64x2xbf16> to memref<32x16x2xbf16, strided<[128, 2, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg4, 0] [16, 32, 2] [1, 1, 1] : memref<64x128x2xbf16> to memref<16x32x2xbf16, strided<[256, 2, 1], offset: ?>>
            %7 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} : memref<32x16x2xbf16, strided<[128, 2, 1], offset: ?>>, vector<16x16x2xbf16>
            %8 = vector.transfer_read %subview_0[%c16, %c0, %c0], %1 {in_bounds = [true, true, true]} : memref<32x16x2xbf16, strided<[128, 2, 1], offset: ?>>, vector<16x16x2xbf16>
            %9 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} : memref<16x32x2xbf16, strided<[256, 2, 1], offset: ?>>, vector<16x16x2xbf16>
            %10 = vector.transfer_read %subview_1[%c0, %c16, %c0], %1 {in_bounds = [true, true, true]} : memref<16x32x2xbf16, strided<[256, 2, 1], offset: ?>>, vector<16x16x2xbf16>
            %11 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %7, %9, %arg6 {unroll_shape = array<i64: 2, 16, 16, 16>} : vector<16x16x2xbf16>, vector<16x16x2xbf16> into vector<16x16xf32>
            %12 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %7, %10, %arg7 {unroll_shape = array<i64: 2, 16, 16, 16>} : vector<16x16x2xbf16>, vector<16x16x2xbf16> into vector<16x16xf32>
            %13 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %8, %9, %arg8 {unroll_shape = array<i64: 2, 16, 16, 16>} : vector<16x16x2xbf16>, vector<16x16x2xbf16> into vector<16x16xf32>
            %14 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %8, %10, %arg9 {unroll_shape = array<i64: 2, 16, 16, 16>} : vector<16x16x2xbf16>, vector<16x16x2xbf16> into vector<16x16xf32>
            scf.yield %11, %12, %13, %14 : vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>
          }
          vector.transfer_write %6#3, %subview[%c16, %c16] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[128, 1], offset: ?>>
          vector.transfer_write %6#2, %subview[%c16, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[128, 1], offset: ?>>
          vector.transfer_write %6#1, %subview[%c0, %c16] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[128, 1], offset: ?>>
          vector.transfer_write %6#0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[128, 1], offset: ?>>
        }
      }
      %alloc = memref.alloc() : memref<64x128xf32>
      memref.copy %arg2, %alloc : memref<64x128xf32> to memref<64x128xf32>
      return %alloc : memref<64x128xf32>
    }
  }

// CHECK-LABEL: @batch_amx
// CHECK: amx.tile_mulf
// CHECK-NOT: vector.contract


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.amx.vector_contract_to_packed_type_tiled_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3)>

  module {
    func.func @matmul_amx(%arg0: memref<16x64x64x2xbf16>, %arg1: memref<16x64x128x2xbf16>, %arg2: memref<16x64x128xf32>) -> memref<16x64x128xf32> attributes {dlti.target_system_spec = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"reg_gemm_unroll" = [16, 16, 16]>>} {
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
          scf.for %arg5 = %c0 to %c16 step %c1 {
            %subview = memref.subview %arg2[%arg5, %arg3, %arg4] [1, 32, 32] [1, 1, 1] : memref<16x64x128xf32> to memref<1x32x32xf32, strided<[8192, 128, 1], offset: ?>>
            %2 = vector.transfer_read %subview[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x32x32xf32, strided<[8192, 128, 1], offset: ?>>, vector<1x16x16xf32>
            %3 = vector.transfer_read %subview[%c0, %c0, %c16], %0 {in_bounds = [true, true, true]} : memref<1x32x32xf32, strided<[8192, 128, 1], offset: ?>>, vector<1x16x16xf32>
            %4 = vector.transfer_read %subview[%c0, %c16, %c0], %0 {in_bounds = [true, true, true]} : memref<1x32x32xf32, strided<[8192, 128, 1], offset: ?>>, vector<1x16x16xf32>
            %5 = vector.transfer_read %subview[%c0, %c16, %c16], %0 {in_bounds = [true, true, true]} : memref<1x32x32xf32, strided<[8192, 128, 1], offset: ?>>, vector<1x16x16xf32>
            %6:4 = scf.for %arg6 = %c0 to %c64 step %c16 iter_args(%arg7 = %2, %arg8 = %3, %arg9 = %4, %arg10 = %5) -> (vector<1x16x16xf32>, vector<1x16x16xf32>, vector<1x16x16xf32>, vector<1x16x16xf32>) {
              %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg6, 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<16x64x64x2xbf16> to memref<1x32x16x2xbf16, strided<[8192, 128, 2, 1], offset: ?>>
              %subview_1 = memref.subview %arg1[%arg5, %arg6, %arg4, 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<16x64x128x2xbf16> to memref<1x16x32x2xbf16, strided<[16384, 256, 2, 1], offset: ?>>
              %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} : memref<1x32x16x2xbf16, strided<[8192, 128, 2, 1], offset: ?>>, vector<1x16x16x2xbf16>
              %8 = vector.transfer_read %subview_0[%c0, %c16, %c0, %c0], %1 {in_bounds = [true, true, true, true]} : memref<1x32x16x2xbf16, strided<[8192, 128, 2, 1], offset: ?>>, vector<1x16x16x2xbf16>
              %9 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} : memref<1x16x32x2xbf16, strided<[16384, 256, 2, 1], offset: ?>>, vector<1x16x16x2xbf16>
              %10 = vector.transfer_read %subview_1[%c0, %c0, %c16, %c0], %1 {in_bounds = [true, true, true, true]} : memref<1x16x32x2xbf16, strided<[16384, 256, 2, 1], offset: ?>>, vector<1x16x16x2xbf16>
              %11 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %7, %9, %arg7 {unroll_shape = array<i64: 1, 2, 16, 16, 16>} : vector<1x16x16x2xbf16>, vector<1x16x16x2xbf16> into vector<1x16x16xf32>
              %12 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %7, %10, %arg8 {unroll_shape = array<i64: 1, 2, 16, 16, 16>} : vector<1x16x16x2xbf16>, vector<1x16x16x2xbf16> into vector<1x16x16xf32>
              %13 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %8, %9, %arg9 {unroll_shape = array<i64: 1, 2, 16, 16, 16>} : vector<1x16x16x2xbf16>, vector<1x16x16x2xbf16> into vector<1x16x16xf32>
              %14 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %8, %10, %arg10 {unroll_shape = array<i64: 1, 2, 16, 16, 16>} : vector<1x16x16x2xbf16>, vector<1x16x16x2xbf16> into vector<1x16x16xf32>
              scf.yield %11, %12, %13, %14 : vector<1x16x16xf32>, vector<1x16x16xf32>, vector<1x16x16xf32>, vector<1x16x16xf32>
            }
            vector.transfer_write %6#3, %subview[%c0, %c16, %c16] {in_bounds = [true, true, true]} : vector<1x16x16xf32>, memref<1x32x32xf32, strided<[8192, 128, 1], offset: ?>>
            vector.transfer_write %6#2, %subview[%c0, %c16, %c0] {in_bounds = [true, true, true]} : vector<1x16x16xf32>, memref<1x32x32xf32, strided<[8192, 128, 1], offset: ?>>
            vector.transfer_write %6#1, %subview[%c0, %c0, %c16] {in_bounds = [true, true, true]} : vector<1x16x16xf32>, memref<1x32x32xf32, strided<[8192, 128, 1], offset: ?>>
            vector.transfer_write %6#0, %subview[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x16x16xf32>, memref<1x32x32xf32, strided<[8192, 128, 1], offset: ?>>
          }
        }
      }
      %alloc = memref.alloc() : memref<16x64x128xf32>
      memref.copy %arg2, %alloc : memref<16x64x128xf32> to memref<16x64x128xf32>
      return %alloc : memref<16x64x128xf32>
    }
  }

// CHECK-LABEL: @matmul_amx
// CHECK: amx.tile_mulf
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.amx.vector_contract_to_packed_type_tiled_dot_product
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
func.func @amx(
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

// CHECK-LABEL: @amx
// CHECK: amx.tile_mulf
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.amx.vector_contract_to_packed_type_tiled_dot_product
    } : !transform.any_op
    transform.yield
  }
}

