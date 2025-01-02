// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
  memref.global "private" constant @__constant_24x64x64xf32 : memref<24x64x64xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @simple_gemm(%arg0: memref<8x24x32x64xf32>) -> memref<8x24x32x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x64xf32>
    %c1 = arith.constant 1 : index
    %c24 = arith.constant 24 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_24x64x64xf32 : memref<24x64x64xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x24x32x64xf32>
    scf.forall (%arg1, %arg2) in (8, 24) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x64xf32>, memref<32x64xf32, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 24, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c32 step %c4 {
        scf.for %arg4 = %c0 to %c64 step %c64 {
          %subview_2 = memref.subview %subview[%arg3, %arg4] [4, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<4x64xf32, strided<[64, 1], offset: ?>>
          %1 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x64xf32, strided<[64, 1], offset: ?>>, vector<4x64xf32>
          %2 = scf.for %arg5 = %c0 to %c24 step %c1 iter_args(%arg6 = %1) -> (vector<4x64xf32>) {
            %3 = scf.for %arg7 = %c0 to %c64 step %c1 iter_args(%arg8 = %arg6) -> (vector<4x64xf32>) {
              %subview_3 = memref.subview %subview_1[%arg5, %arg3, %arg7] [1, 4, 1] [1, 1, 1] : memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>> to memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
              %subview_4 = memref.subview %0[%arg5, %arg7, %arg4] [1, 1, 64] [1, 1, 1] : memref<24x64x64xf32> to memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>
              %4 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>, vector<1x4x1xf32>
              %5 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<1x1x64xf32>
              %6 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x4x1xf32>, vector<1x1x64xf32> into vector<4x64xf32>
              scf.yield %6 : vector<4x64xf32>
            }
            scf.yield %3 : vector<4x64xf32>
          }
          vector.transfer_write %2, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, memref<4x64xf32, strided<[64, 1], offset: ?>>
        }
      }
    }
    return %alloc : memref<8x24x32x64xf32>
  }

// CHECK-LABEL:   memref.global "private" constant @__constant_24x64x64xf32 : memref<24x64x64xf32> = dense<1.000000e+00> {alignment = 64 : i64}

// CHECK-LABEL:   func.func @simple_gemm(
// CHECK-SAME:                           %[[VAL_0:.*]]: memref<8x24x32x64xf32>) -> memref<8x24x32x64xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<0.000000e+00> : vector<32x64xf32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 24 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_10:.*]] = memref.get_global @__constant_24x64x64xf32 : memref<24x64x64xf32>
// CHECK:           %[[VAL_11:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x24x32x64xf32>
// CHECK:           scf.forall (%[[VAL_12:.*]], %[[VAL_13:.*]]) in (8, 24) {
// CHECK:             %[[VAL_14:.*]] = memref.subview %[[VAL_11]]{{\[}}%[[VAL_12]], %[[VAL_13]], 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK:             vector.transfer_write %[[VAL_3]], %[[VAL_14]]{{\[}}%[[VAL_9]], %[[VAL_9]]] {in_bounds = [true, true]} : vector<32x64xf32>, memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK:             %[[VAL_15:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_12]], 0, 0, 0] [1, 24, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_16:.*]] = %[[VAL_9]] to %[[VAL_8]] step %[[VAL_7]] {
// CHECK:               scf.for %[[VAL_17:.*]] = %[[VAL_9]] to %[[VAL_6]] step %[[VAL_6]] {
// CHECK:                 %[[VAL_18:.*]] = memref.subview %[[VAL_14]]{{\[}}%[[VAL_16]], %[[VAL_17]]] [4, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<4x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                 %[[VAL_19:.*]] = memref.subview %[[VAL_18]][0, 0] [1, 64] [1, 1] : memref<4x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                 %[[VAL_20:.*]] = memref.subview %[[VAL_18]][1, 0] [1, 64] [1, 1] : memref<4x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                 %[[VAL_21:.*]] = memref.subview %[[VAL_18]][2, 0] [1, 64] [1, 1] : memref<4x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                 %[[VAL_22:.*]] = memref.subview %[[VAL_18]][3, 0] [1, 64] [1, 1] : memref<4x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                 %[[VAL_23:.*]] = vector.load %[[VAL_19]]{{\[}}%[[VAL_9]], %[[VAL_9]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:                 %[[VAL_24:.*]] = vector.load %[[VAL_20]]{{\[}}%[[VAL_9]], %[[VAL_9]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:                 %[[VAL_25:.*]] = vector.load %[[VAL_21]]{{\[}}%[[VAL_9]], %[[VAL_9]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:                 %[[VAL_26:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_9]], %[[VAL_9]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:                 %[[VAL_27:.*]]:4 = scf.for %[[VAL_28:.*]] = %[[VAL_9]] to %[[VAL_5]] step %[[VAL_4]] iter_args(%[[VAL_29:.*]] = %[[VAL_23]], %[[VAL_30:.*]] = %[[VAL_24]], %[[VAL_31:.*]] = %[[VAL_25]], %[[VAL_32:.*]] = %[[VAL_26]]) -> (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>) {
// CHECK:                   %[[VAL_33:.*]]:4 = scf.for %[[VAL_34:.*]] = %[[VAL_9]] to %[[VAL_6]] step %[[VAL_4]] iter_args(%[[VAL_35:.*]] = %[[VAL_29]], %[[VAL_36:.*]] = %[[VAL_30]], %[[VAL_37:.*]] = %[[VAL_31]], %[[VAL_38:.*]] = %[[VAL_32]]) -> (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>) {
// CHECK:                     %[[VAL_39:.*]] = memref.subview %[[VAL_15]]{{\[}}%[[VAL_28]], %[[VAL_16]], %[[VAL_34]]] [1, 4, 1] [1, 1, 1] : memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>> to memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_40:.*]] = memref.load %[[VAL_39]]{{\[}}%[[VAL_9]], %[[VAL_9]], %[[VAL_9]]] : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_41:.*]] = vector.broadcast %[[VAL_40]] : f32 to vector<64xf32>
// CHECK:                     %[[VAL_42:.*]] = memref.load %[[VAL_39]]{{\[}}%[[VAL_9]], %[[VAL_4]], %[[VAL_9]]] : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_43:.*]] = vector.broadcast %[[VAL_42]] : f32 to vector<64xf32>
// CHECK:                     %[[VAL_44:.*]] = memref.load %[[VAL_39]]{{\[}}%[[VAL_9]], %[[VAL_2]], %[[VAL_9]]] : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_45:.*]] = vector.broadcast %[[VAL_44]] : f32 to vector<64xf32>
// CHECK:                     %[[VAL_46:.*]] = memref.load %[[VAL_39]]{{\[}}%[[VAL_9]], %[[VAL_1]], %[[VAL_9]]] : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_47:.*]] = vector.broadcast %[[VAL_46]] : f32 to vector<64xf32>
// CHECK:                     %[[VAL_48:.*]] = memref.subview %[[VAL_10]]{{\[}}%[[VAL_28]], %[[VAL_34]], %[[VAL_17]]] [1, 1, 64] [1, 1, 1] : memref<24x64x64xf32> to memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_49:.*]] = vector.load %[[VAL_48]]{{\[}}%[[VAL_9]], %[[VAL_9]], %[[VAL_9]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<64xf32>
// CHECK:                     %[[VAL_50:.*]] = vector.fma %[[VAL_41]], %[[VAL_49]], %[[VAL_35]] : vector<64xf32>
// CHECK:                     %[[VAL_51:.*]] = vector.fma %[[VAL_43]], %[[VAL_49]], %[[VAL_36]] : vector<64xf32>
// CHECK:                     %[[VAL_52:.*]] = vector.fma %[[VAL_45]], %[[VAL_49]], %[[VAL_37]] : vector<64xf32>
// CHECK:                     %[[VAL_53:.*]] = vector.fma %[[VAL_47]], %[[VAL_49]], %[[VAL_38]] : vector<64xf32>
// CHECK:                     scf.yield %[[VAL_50]], %[[VAL_51]], %[[VAL_52]], %[[VAL_53]] : vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_54:.*]]#0, %[[VAL_54]]#1, %[[VAL_54]]#2, %[[VAL_54]]#3 : vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>
// CHECK:                 }
// CHECK:                 vector.store %[[VAL_55:.*]]#0, %[[VAL_19]]{{\[}}%[[VAL_9]], %[[VAL_9]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:                 vector.store %[[VAL_55]]#1, %[[VAL_20]]{{\[}}%[[VAL_9]], %[[VAL_9]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:                 vector.store %[[VAL_55]]#2, %[[VAL_21]]{{\[}}%[[VAL_9]], %[[VAL_9]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:                 vector.store %[[VAL_55]]#3, %[[VAL_22]]{{\[}}%[[VAL_9]], %[[VAL_9]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_11]] : memref<8x24x32x64xf32>
// CHECK:         }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %0 {
        transform.apply_patterns.vector.contract_to_fma
      } : !transform.any_op
      transform.yield
    }
  }
