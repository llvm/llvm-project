// RUN: mlir-opt %s -pass-pipeline='func.func(scf-parallel-loop-tiling{parallel-loop-tile-sizes=1,4})' -split-input-file | FileCheck %s

func @parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index,
                    %arg3 : index, %arg4 : index, %arg5 : index,
		    %A: memref<?x?xf32>, %B: memref<?x?xf32>,
                    %C: memref<?x?xf32>, %result: memref<?x?xf32>) {
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3) step (%arg4, %arg5) {
    %B_elem = memref.load %B[%i0, %i1] : memref<?x?xf32>
    %C_elem = memref.load %C[%i0, %i1] : memref<?x?xf32>
    %sum_elem = arith.addf %B_elem, %C_elem : f32
    memref.store %sum_elem, %result[%i0, %i1] : memref<?x?xf32>
  }
  return
}

// CHECK:       #map = affine_map<(d0, d1, d2) -> (d0, d1 - d2)>
// CHECK-LABEL:   func @parallel_loop(
// CHECK-SAME:                        [[ARG1:%.*]]: index, [[ARG2:%.*]]: index, [[ARG3:%.*]]: index, [[ARG4:%.*]]: index, [[ARG5:%.*]]: index, [[ARG6:%.*]]: index, [[ARG7:%.*]]: memref<?x?xf32>, [[ARG8:%.*]]: memref<?x?xf32>, [[ARG9:%.*]]: memref<?x?xf32>, [[ARG10:%.*]]: memref<?x?xf32>) {
// CHECK:           [[C0:%.*]] = arith.constant 0 : index
// CHECK:           [[C1:%.*]] = arith.constant 1 : index
// CHECK:           [[C4:%.*]] = arith.constant 4 : index
// CHECK:           [[V1:%.*]] = arith.muli [[ARG5]], [[C1]] : index
// CHECK:           [[V2:%.*]] = arith.muli [[ARG6]], [[C4]] : index
// CHECK:           scf.parallel ([[V3:%.*]], [[V4:%.*]]) = ([[ARG1]], [[ARG2]]) to ([[ARG3]], [[ARG4]]) step ([[V1]], [[V2]]) {
// CHECK:             [[V5:%.*]] = affine.min #map([[V1]], [[ARG3]], [[V3]])
// CHECK:             [[V6:%.*]] = affine.min #map([[V2]], [[ARG4]], [[V4]])
// CHECK:             scf.parallel ([[V7:%.*]], [[V8:%.*]]) = ([[C0]], [[C0]]) to ([[V5]], [[V6]]) step ([[ARG5]], [[ARG6]]) {
// CHECK:               [[V9:%.*]] = arith.addi [[V7]], [[V3]] : index
// CHECK:               [[V10:%.*]] = arith.addi [[V8]], [[V4]] : index
// CHECK:               [[V11:%.*]] = memref.load [[ARG8]]{{\[}}[[V9]], [[V10]]] : memref<?x?xf32>
// CHECK:               [[V12:%.*]] = memref.load [[ARG9]]{{\[}}[[V9]], [[V10]]] : memref<?x?xf32>
// CHECK:               [[V13:%.*]] = arith.addf [[V11]], [[V12]] : f32
// CHECK:               memref.store [[V13]], [[ARG10]]{{\[}}[[V9]], [[V10]]] : memref<?x?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return

// -----

func @static_loop_with_step() {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c22 = arith.constant 22 : index
  %c24 = arith.constant 24 : index
  scf.parallel (%i0, %i1) = (%c0, %c0) to (%c22, %c24) step (%c3, %c3) {
  }
  return
}

// CHECK-LABEL:   func @static_loop_with_step() {
// CHECK:           [[C0:%.*]] = arith.constant 0 : index
// CHECK:           [[C3:%.*]] = arith.constant 3 : index
// CHECK:           [[C22:%.*]] = arith.constant 22 : index
// CHECK:           [[C24:%.*]] = arith.constant 24 : index
// CHECK:           [[C0_1:%.*]] = arith.constant 0 : index
// CHECK:           [[C1:%.*]] = arith.constant 1 : index
// CHECK:           [[C4:%.*]] = arith.constant 4 : index
// CHECK:           [[V1:%.*]] = arith.muli [[C3]], [[C1]] : index
// CHECK:           [[V2:%.*]] = arith.muli [[C3]], [[C4]] : index
// CHECK:           scf.parallel ([[V3:%.*]], [[V4:%.*]]) = ([[C0]], [[C0]]) to ([[C22]], [[C24]]) step ([[V1]], [[V2]]) {
// CHECK:             scf.parallel ([[V5:%.*]], [[V6:%.*]]) = ([[C0_1]], [[C0_1]]) to ([[V1]], [[V2]]) step ([[C3]], [[C3]]) {
// CHECK:               = arith.addi [[V5]], [[V3]] : index
// CHECK:               = arith.addi [[V6]], [[V4]] : index
// CHECK:             }
// CHECK:           }
// CHECK:           return

// -----

func @tile_nested_innermost() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.parallel (%k, %l) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    }
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
  }
  return
}

// CHECK-LABEL:   func @tile_nested_innermost() {
// CHECK:           [[C2:%.*]] = arith.constant 2 : index
// CHECK:           [[C0:%.*]] = arith.constant 0 : index
// CHECK:           [[C1:%.*]] = arith.constant 1 : index
// CHECK:           scf.parallel ([[V1:%.*]], [[V2:%.*]]) = ([[C0]], [[C0]]) to ([[C2]], [[C2]]) step ([[C1]], [[C1]]) {
// CHECK:             [[C0_1:%.*]] = arith.constant 0 : index
// CHECK:             [[C1_1:%.*]] = arith.constant 1 : index
// CHECK:             [[C4:%.*]] = arith.constant 4 : index
// CHECK:             [[V3:%.*]] = arith.muli [[C1]], [[C1_1]] : index
// CHECK:             [[V4:%.*]] = arith.muli [[C1]], [[C4]] : index
// CHECK:             scf.parallel ([[V5:%.*]], [[V6:%.*]]) = ([[C0]], [[C0]]) to ([[C2]], [[C2]]) step ([[V3]], [[V4]]) {
// CHECK:               [[V7:%.*]] = affine.min #map([[V4]], [[C2]], [[V6]])
// CHECK:               scf.parallel ([[V8:%.*]], [[V9:%.*]]) = ([[C0_1]], [[C0_1]]) to ([[V3]], [[V7]]) step ([[C1]], [[C1]]) {
// CHECK:                 = arith.addi [[V8]], [[V5]] : index
// CHECK:                 = arith.addi [[V9]], [[V6]] : index
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[C0_2:%.*]] = arith.constant 0 : index
// CHECK:           [[C1_2:%.*]] = arith.constant 1 : index
// CHECK:           [[C4_1:%.*]] = arith.constant 4 : index
// CHECK:           [[V10:%.*]] = arith.muli [[C1]], [[C1_2]] : index
// CHECK:           [[V11:%.*]] = arith.muli [[C1]], [[C4_1]] : index
// CHECK:           scf.parallel ([[V12:%.*]], [[V13:%.*]]) = ([[C0]], [[C0]]) to ([[C2]], [[C2]]) step ([[V10]], [[V11]]) {
// CHECK:             [[V14:%.*]] = affine.min #map([[V11]], [[C2]], [[V13]])
// CHECK:             scf.parallel ([[V15:%.*]], [[V16:%.*]]) = ([[C0_2]], [[C0_2]]) to ([[V10]], [[V14]]) step ([[C1]], [[C1]]) {
// CHECK:               = arith.addi [[V15]], [[V12]] : index
// CHECK:               = arith.addi [[V16]], [[V13]] : index
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// -----

func @tile_nested_in_non_ploop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c2 step %c1 {
      scf.parallel (%k, %l) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
      }
    }
  }
  return
}

// CHECK-LABEL: func @tile_nested_in_non_ploop
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.parallel
// CHECK:               scf.parallel
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
