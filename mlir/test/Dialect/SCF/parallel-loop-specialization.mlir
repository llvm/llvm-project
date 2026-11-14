// RUN: mlir-opt %s -scf-parallel-loop-specialization -split-input-file | FileCheck %s

#map0 = affine_map<()[s0, s1] -> (1024, s0 - s1)>
#map1 = affine_map<()[s0, s1] -> (64, s0 - s1)>

func.func @parallel_loop(%outer_i0: index, %outer_i1: index, %A: memref<?x?xf32>, %B: memref<?x?xf32>,
                    %C: memref<?x?xf32>, %result: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = memref.dim %A, %c0 : memref<?x?xf32>
  %d1 = memref.dim %A, %c1 : memref<?x?xf32>
  %b0 = affine.min #map0()[%d0, %outer_i0]
  %b1 = affine.min #map1()[%d1, %outer_i1]
  scf.parallel (%i0, %i1) = (%c0, %c0) to (%b0, %b1) step (%c1, %c1) {
    %B_elem = memref.load %B[%i0, %i1] : memref<?x?xf32>
    %C_elem = memref.load %C[%i0, %i1] : memref<?x?xf32>
    %sum_elem = arith.addf %B_elem, %C_elem : f32
    memref.store %sum_elem, %result[%i0, %i1] : memref<?x?xf32>
  }
  return
}

// CHECK-LABEL:   func @parallel_loop(
// CHECK-SAME:                        [[VAL_0:%.*]]: index, [[VAL_1:%.*]]: index, [[VAL_2:%.*]]: memref<?x?xf32>, [[VAL_3:%.*]]: memref<?x?xf32>, [[VAL_4:%.*]]: memref<?x?xf32>, [[VAL_5:%.*]]: memref<?x?xf32>) {
// CHECK:           [[VAL_6:%.*]] = arith.constant 0 : index
// CHECK:           [[VAL_7:%.*]] = arith.constant 1 : index
// CHECK:           [[VAL_8:%.*]] = memref.dim [[VAL_2]], [[VAL_6]] : memref<?x?xf32>
// CHECK:           [[VAL_9:%.*]] = memref.dim [[VAL_2]], [[VAL_7]] : memref<?x?xf32>
// CHECK:           [[VAL_10:%.*]] = affine.min #{{.*}}(){{\[}}[[VAL_8]], [[VAL_0]]]
// CHECK:           [[VAL_11:%.*]] = affine.min #{{.*}}(){{\[}}[[VAL_9]], [[VAL_1]]]
// CHECK:           [[VAL_12:%.*]] = arith.constant 1024 : index
// CHECK:           [[VAL_13:%.*]] = arith.cmpi eq, [[VAL_10]], [[VAL_12]] : index
// CHECK:           [[VAL_14:%.*]] = arith.constant 64 : index
// CHECK:           [[VAL_15:%.*]] = arith.cmpi eq, [[VAL_11]], [[VAL_14]] : index
// CHECK:           [[VAL_16:%.*]] = arith.andi [[VAL_13]], [[VAL_15]] : i1
// CHECK:           scf.if [[VAL_16]] {
// CHECK:             scf.parallel ([[VAL_17:%.*]], [[VAL_18:%.*]]) = ([[VAL_6]], [[VAL_6]]) to ([[VAL_12]], [[VAL_14]]) step ([[VAL_7]], [[VAL_7]]) {
// CHECK:               memref.store
// CHECK:             }
// CHECK:           } else {
// CHECK:             scf.parallel ([[VAL_22:%.*]], [[VAL_23:%.*]]) = ([[VAL_6]], [[VAL_6]]) to ([[VAL_10]], [[VAL_11]]) step ([[VAL_7]], [[VAL_7]]) {
// CHECK:               memref.store
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// -----

#map = affine_map<()[s0] -> (8, s0)>

func.func @parallel_loop_with_result(%ub: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %bound = affine.min #map()[%ub]
  %result = scf.parallel (%i) = (%c0) to (%bound) step (%c1)
      init (%init) -> i32 {
    %value = arith.index_cast %i : index to i32
    scf.reduce(%value : i32) {
    ^bb0(%lhs: i32, %rhs: i32):
      %sum = arith.addi %lhs, %rhs : i32
      scf.reduce.return %sum : i32
    }
  }
  return %result : i32
}

// CHECK-LABEL: func.func @parallel_loop_with_result
// CHECK: %[[IF_RESULT:.*]] = scf.if {{.*}} -> (i32) {
// CHECK:   %[[THEN_RESULT:.*]] = scf.parallel
// CHECK:   scf.yield %[[THEN_RESULT]] : i32
// CHECK: } else {
// CHECK:   %[[ELSE_RESULT:.*]] = scf.parallel
// CHECK:   scf.yield %[[ELSE_RESULT]] : i32
// CHECK: }
// CHECK: return %[[IF_RESULT]] : i32
