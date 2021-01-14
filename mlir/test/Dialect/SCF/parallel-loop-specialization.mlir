// RUN: mlir-opt %s -parallel-loop-specialization -split-input-file | FileCheck %s

#map0 = affine_map<()[s0, s1] -> (1024, s0 - s1)>
#map1 = affine_map<()[s0, s1] -> (64, s0 - s1)>

func @parallel_loop(%outer_i0: index, %outer_i1: index, %A: memref<?x?xf32>, %B: memref<?x?xf32>,
                    %C: memref<?x?xf32>, %result: memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %d0 = dim %A, %c0 : memref<?x?xf32>
  %d1 = dim %A, %c1 : memref<?x?xf32>
  %b0 = affine.min #map0()[%d0, %outer_i0]
  %b1 = affine.min #map1()[%d1, %outer_i1]
  scf.parallel (%i0, %i1) = (%c0, %c0) to (%b0, %b1) step (%c1, %c1) {
    %B_elem = load %B[%i0, %i1] : memref<?x?xf32>
    %C_elem = load %C[%i0, %i1] : memref<?x?xf32>
    %sum_elem = addf %B_elem, %C_elem : f32
    store %sum_elem, %result[%i0, %i1] : memref<?x?xf32>
  }
  return
}

// CHECK-LABEL:   func @parallel_loop(
// CHECK-SAME:                        [[VAL_0:%.*]]: index, [[VAL_1:%.*]]: index, [[VAL_2:%.*]]: memref<?x?xf32>, [[VAL_3:%.*]]: memref<?x?xf32>, [[VAL_4:%.*]]: memref<?x?xf32>, [[VAL_5:%.*]]: memref<?x?xf32>) {
// CHECK:           [[VAL_6:%.*]] = constant 0 : index
// CHECK:           [[VAL_7:%.*]] = constant 1 : index
// CHECK:           [[VAL_8:%.*]] = dim [[VAL_2]], [[VAL_6]] : memref<?x?xf32>
// CHECK:           [[VAL_9:%.*]] = dim [[VAL_2]], [[VAL_7]] : memref<?x?xf32>
// CHECK:           [[VAL_10:%.*]] = affine.min #map0(){{\[}}[[VAL_8]], [[VAL_0]]]
// CHECK:           [[VAL_11:%.*]] = affine.min #map1(){{\[}}[[VAL_9]], [[VAL_1]]]
// CHECK:           [[VAL_12:%.*]] = constant 1024 : index
// CHECK:           [[VAL_13:%.*]] = cmpi eq, [[VAL_10]], [[VAL_12]] : index
// CHECK:           [[VAL_14:%.*]] = constant 64 : index
// CHECK:           [[VAL_15:%.*]] = cmpi eq, [[VAL_11]], [[VAL_14]] : index
// CHECK:           [[VAL_16:%.*]] = and [[VAL_13]], [[VAL_15]] : i1
// CHECK:           scf.if [[VAL_16]] {
// CHECK:             scf.parallel ([[VAL_17:%.*]], [[VAL_18:%.*]]) = ([[VAL_6]], [[VAL_6]]) to ([[VAL_12]], [[VAL_14]]) step ([[VAL_7]], [[VAL_7]]) {
// CHECK:               store
// CHECK:             }
// CHECK:           } else {
// CHECK:             scf.parallel ([[VAL_22:%.*]], [[VAL_23:%.*]]) = ([[VAL_6]], [[VAL_6]]) to ([[VAL_10]], [[VAL_11]]) step ([[VAL_7]], [[VAL_7]]) {
// CHECK:               store
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
