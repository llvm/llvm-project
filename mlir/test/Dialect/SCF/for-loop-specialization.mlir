// RUN: mlir-opt %s -for-loop-specialization -split-input-file | FileCheck %s

#map0 = affine_map<()[s0, s1] -> (1024, s0 - s1)>
#map1 = affine_map<()[s0, s1] -> (64, s0 - s1)>

func @for(%outer: index, %A: memref<?xf32>, %B: memref<?xf32>,
          %C: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %d0 = memref.dim %A, %c0 : memref<?xf32>
  %b0 = affine.min #map0()[%d0, %outer]
  scf.for %i0 = %c0 to %b0 step %c1 {
    %B_elem = memref.load %B[%i0] : memref<?xf32>
    %C_elem = memref.load %C[%i0] : memref<?xf32>
    %sum_elem = addf %B_elem, %C_elem : f32
    memref.store %sum_elem, %result[%i0] : memref<?xf32>
  }
  return
}

// CHECK-LABEL:   func @for(
// CHECK-SAME:              [[ARG0:%.*]]: index, [[ARG1:%.*]]: memref<?xf32>, [[ARG2:%.*]]: memref<?xf32>, [[ARG3:%.*]]: memref<?xf32>, [[ARG4:%.*]]: memref<?xf32>) {
// CHECK:           [[CST_0:%.*]] = constant 0 : index
// CHECK:           [[CST_1:%.*]] = constant 1 : index
// CHECK:           [[DIM_0:%.*]] = memref.dim [[ARG1]], [[CST_0]] : memref<?xf32>
// CHECK:           [[MIN:%.*]] = affine.min #map(){{\[}}[[DIM_0]], [[ARG0]]]
// CHECK:           [[CST_1024:%.*]] = constant 1024 : index
// CHECK:           [[PRED:%.*]] = cmpi eq, [[MIN]], [[CST_1024]] : index
// CHECK:           scf.if [[PRED]] {
// CHECK:             scf.for [[IDX0:%.*]] = [[CST_0]] to [[CST_1024]] step [[CST_1]] {
// CHECK:               memref.store
// CHECK:             }
// CHECK:           } else {
// CHECK:             scf.for [[IDX0:%.*]] = [[CST_0]] to [[MIN]] step [[CST_1]] {
// CHECK:               memref.store
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
