// RUN: mlir-opt -test-extract-fixed-outer-loops %s | FileCheck %s 

// COMMON-LABEL: @no_crash
func.func @no_crash(%arg0: memref<?x?xf32>) {
  // CHECK: %[[c2:.*]] = arith.constant 2 : index
  %c2 = arith.constant 2 : index
  // CHECK: %[[c44:.*]] = arith.constant 44 : index
  %c44 = arith.constant 44 : index
  // CHECK: %[[c1:.*]] = arith.constant 1 : index  
  %c1 = arith.constant 1 : index
  // CHECK: scf.for %[[i:.*]] = %[[c2]] to %[[c44]] step %[[c1]]
  scf.for %i = %c2 to %c44 step %c1 {
    // CHECK: scf.for %[[j:.*]] = %[[c1]] to %[[c44]] step %[[c2]]
    scf.for %j = %c1 to %c44 step %c2 {
      memref.load %arg0[%i, %j]: memref<?x?xf32>
    }
  }
  return
}
