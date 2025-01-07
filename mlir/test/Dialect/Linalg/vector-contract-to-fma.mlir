// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#mapTransposeB = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

  func.func @transpose_matrix_no_conversion_to_fma(%arg0: memref<16x32x128xf32>, %arg1: memref<16x128x64xf32>, %arg2: memref<32x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index

    scf.for %arg5 = %c0 to %c32 step %c4 {
      scf.for %arg6 = %c0 to %c128 step %c64 {
        %subview_2 = memref.subview %arg2[%arg5, %arg6] [4, 64] [1, 1] : memref<32x64xf32> to memref<4x64xf32, strided<[64, 1], offset: ?>>
        %2 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x64xf32, strided<[64, 1], offset: ?>>, vector<4x64xf32>
        %con = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%argcon = %2) -> vector<4x64xf32> {
          %con1 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%argcon1 = %argcon) -> vector<4x64xf32> {
            %subview_3 = memref.subview %arg0[%arg7, %arg5, %arg8] [1, 4, 1] [1, 1, 1] : memref<16x32x128xf32> to memref<1x4x1xf32, strided<[4096, 128, 1], offset: ?>>
            %subview_4 = memref.subview %arg1[%arg7, %arg8, %arg6] [1, 1, 64] [1, 1, 1] : memref<16x128x64xf32> to memref<1x1x64xf32, strided<[8192, 64, 1], offset: ?>>
            %0 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x1xf32, strided<[4096, 128, 1], offset: ?>>, vector<1x4x1xf32>
            %1 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {permutation_map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>, in_bounds = [true, true, true]} : memref<1x1x64xf32, strided<[8192, 64, 1], offset: ?>>, vector<1x64x1xf32>
            %3 = vector.contract {indexing_maps = [#map, #mapTransposeB, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %argcon1 : vector<1x4x1xf32>, vector<1x64x1xf32> into vector<4x64xf32>
            scf.yield %3 : vector<4x64xf32>
          }
          scf.yield %con1 : vector<4x64xf32>
        }
        vector.transfer_write %con, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, memref<4x64xf32, strided<[64, 1], offset: ?>>
      }
    }
    return
  }

// CHECK-NOT: vector.fma

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %0 {
        transform.apply_patterns.vector.contract_to_fma
      } : !transform.any_op
      transform.yield
    }
  }
