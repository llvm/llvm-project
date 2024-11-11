// RUN: mlir-opt %s -split-input-file -affine-parallel-banking="banking-factor=2" | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0 mod 2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0 floordiv 2)>

// CHECK-LABEL:   func.func @parallel_bank_one_dim(
// CHECK:                                     %[[VAL_0:arg0]]: memref<4xf32>,
// CHECK:                                     %[[VAL_1:arg1]]: memref<4xf32>,
// CHECK:                                     %[[VAL_2:arg2]]: memref<4xf32>,
// CHECK:                                     %[[VAL_3:arg3]]: memref<4xf32>) -> (memref<4xf32>, memref<4xf32>) {
// CHECK:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           affine.parallel (%[[VAL_7:.*]]) = (0) to (8) {
// CHECK:             %[[VAL_8:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]])
// CHECK:             %[[VAL_9:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_7]])
// CHECK:             %[[VAL_10:.*]] = scf.index_switch %[[VAL_8]] -> f32
// CHECK:             case 0 {
// CHECK:               %[[VAL_11:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_9]]] : memref<4xf32>
// CHECK:               scf.yield %[[VAL_11]] : f32
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               %[[VAL_12:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_9]]] : memref<4xf32>
// CHECK:               scf.yield %[[VAL_12]] : f32
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_4]] : f32
// CHECK:             }
// CHECK:             %[[VAL_13:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]])
// CHECK:             %[[VAL_14:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_7]])
// CHECK:             %[[VAL_15:.*]] = scf.index_switch %[[VAL_13]] -> f32
// CHECK:             case 0 {
// CHECK:               %[[VAL_16:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_14]]] : memref<4xf32>
// CHECK:               scf.yield %[[VAL_16]] : f32
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               %[[VAL_17:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_14]]] : memref<4xf32>
// CHECK:               scf.yield %[[VAL_17]] : f32
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_4]] : f32
// CHECK:             }
// CHECK:             %[[VAL_18:.*]] = arith.mulf %[[VAL_10]], %[[VAL_15]] : f32
// CHECK:             %[[VAL_19:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]])
// CHECK:             %[[VAL_20:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_7]])
// CHECK:             scf.index_switch %[[VAL_19]]
// CHECK:             case 0 {
// CHECK:               affine.store %[[VAL_18]], %[[VAL_5]]{{\[}}%[[VAL_20]]] : memref<4xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               affine.store %[[VAL_18]], %[[VAL_6]]{{\[}}%[[VAL_20]]] : memref<4xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             default {
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_5]], %[[VAL_6]] : memref<4xf32>, memref<4xf32>
// CHECK:         }
func.func @parallel_bank_one_dim(%arg0: memref<8xf32>, %arg1: memref<8xf32>) -> (memref<8xf32>) {
  %mem = memref.alloc() : memref<8xf32>
  affine.parallel (%i) = (0) to (8) {
    %1 = affine.load %arg0[%i] : memref<8xf32>
    %2 = affine.load %arg1[%i] : memref<8xf32>
    %3 = arith.mulf %1, %2 : f32
    affine.store %3, %mem[%i] : memref<8xf32>
  }
  return %mem : memref<8xf32>
}
