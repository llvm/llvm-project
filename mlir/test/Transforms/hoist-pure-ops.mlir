// RUN: mlir-opt %s -hoist-pure-ops -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @hoist_cast_pos
//  CHECK-SAME:   %[[ARG0:.*]]: memref<10xf32>,
//  CHECK-SAME:   %[[ARG1:.*]]: i1
func.func @hoist_cast_pos(%arg: memref<10xf32>, %arg1: i1) -> (memref<?xf32>) {
  //      CHECK: %[[CAST_0:.*]] = memref.cast %[[ARG0]]
  //      CHECK: %[[CAST_1:.*]] = memref.cast %[[ARG0]]
  // CHECK-NEXT: cf.cond_br %[[ARG1]]
  cf.cond_br %arg1, ^bb1, ^bb2
^bb1:
  %cast = memref.cast %arg : memref<10xf32> to memref<?xf32>
  // CHECK: return %[[CAST_1]]
  return %cast : memref<?xf32>
^bb2:
  %cast1 = memref.cast %arg : memref<10xf32> to memref<?xf32>
  // CHECK: return %[[CAST_0]]
  return %cast1 : memref<?xf32> 
}

// -----

// CHECK-LABEL: func.func @hoist_cast_pos_alloc
//  CHECK-SAME:   %[[ARG0:.*]]: i1
func.func @hoist_cast_pos_alloc(%arg: i1) -> (memref<?xf32>) {
  //      CHECK: %[[ALLOC_0:.*]] = memref.alloc()
  //      CHECK: %[[CAST_0:.*]] = memref.cast %[[ALLOC_0]]
  //      CHECK: %[[CAST_1:.*]] = memref.cast %[[ALLOC_0]]
  // CHECK-NEXT: cf.cond_br %[[ARG0]]
  %alloc = memref.alloc() : memref<10xf32>
  cf.cond_br %arg, ^bb1, ^bb2
^bb1:
  %cast = memref.cast %alloc : memref<10xf32> to memref<?xf32>
  // CHECK: return %[[CAST_1]]
  return %cast : memref<?xf32>
^bb2:
  %cast1 = memref.cast %alloc : memref<10xf32> to memref<?xf32>
  // CHECK: return %[[CAST_0]]
  return %cast1 : memref<?xf32> 
}

// -----

// CHECK-LABEL: func @mult_scf_sum(
//  CHECK-SAME:   %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index
func.func @mult_scf_sum(%arg0: index, %arg1: index, %arg2: index) -> index {
  %c0 = arith.constant 0 : index
  %res0 = scf.for %iv0 = %arg0 to %arg1 step %arg2 iter_args(%sum0 = %c0) -> index {
    %res1 = scf.for %iv1 = %arg0 to %arg1 step %arg2 iter_args(%sum1 = %sum0) -> index {
      %res2 = scf.for %iv2 = %arg0 to %arg1 step %arg2 iter_args(%sum2 = %sum1) -> index {
        %add0 = arith.addi %iv0, %iv1 : index
        %add1 = arith.addi %add0, %iv2 : index
        %add2 = arith.addi %add1, %sum2 : index
        scf.yield %add1 : index
      }
      scf.yield %res2 : index
    }
    scf.yield %res1 : index
  }
  //      CHECK: %[[FOR_0:.*]] = scf.for %[[IV_0:.*]] = %[[ARG0]] to %[[ARG1]] step %[[ARG2]]
  // CHECK-NEXT:   %[[FOR_1:.*]] = scf.for %[[IV_1:.*]] = %[[ARG0]] to %[[ARG1]] step %[[ARG2]]
  // CHECK-NEXT:     %[[ADDI_0:.*]] = arith.addi %[[IV_0]], %[[IV_1]] : index
  // CHECK-NEXT:       %[[FOR_2:.*]] = scf.for %[[IV_3:.*]] = %[[ARG0]] to %[[ARG1]] step %[[ARG2]] iter_args(%[[ITER:.*]] = %{{.*}})
  // CHECK-NEXT:         %[[ADDI_1:.*]] = arith.addi %[[ADDI_0]], %[[IV_3]] : index
  // CHECK-NEXT:         %[[ADDI_2:.*]] = arith.addi %[[ADDI_1]], %[[ITER]] : index
  return %res0 : index
}