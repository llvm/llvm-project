// RUN: mlir-opt %s -test-transform-dialect-interpreter -split-input-file -verify-diagnostics | FileCheck %s

func.func @coalesce_inner() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: scf.for %[[IV0:.+]]
  // CHECK:   scf.for %[[IV1:.+]]
  // CHECK:     scf.for %[[IV2:.+]]
  // CHECK-NOT:   scf.for %[[IV3:.+]]
  scf.for %i = %c0 to %c10 step %c1 {
    scf.for %j = %c0 to %c10 step %c1 {
      scf.for %k = %i to %j step %c1 {
        // Inner loop must have been removed.
        scf.for %l = %i to %j step %c1 {
          arith.addi %i, %j : index
        }
      } {coalesce}
    }
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["scf.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.for">
  %2 = transform.loop.coalesce %1: (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)
}

// -----

func.func @coalesce_outer(%arg1: memref<64x64xf32, 1>, %arg2: memref<64x64xf32, 1>, %arg3: memref<64x64xf32, 1>) attributes {} {
  // CHECK: affine.for %[[IV1:.+]] = 0 to %[[UB:.+]] {
  // CHECK-NOT: affine.for %[[IV2:.+]]
  affine.for %arg4 = 0 to 64 {
    affine.for %arg5 = 0 to 64 {
      // CHECK: %[[IDX0:.+]] = affine.apply #[[MAP0:.+]](%[[IV1]])[%{{.+}}]
      // CHECK: %[[IDX1:.+]] = affine.apply #[[MAP1:.+]](%[[IV1]])[%{{.+}}]
      // CHECK-NEXT: %{{.+}} = affine.load %{{.+}}[%[[IDX1]], %[[IDX0]]] : memref<64x64xf32, 1>
      %0 = affine.load %arg1[%arg4, %arg5] : memref<64x64xf32, 1>
      %1 = affine.load %arg2[%arg4, %arg5] : memref<64x64xf32, 1>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %arg3[%arg4, %arg5] : memref<64x64xf32, 1>
    }
  } {coalesce}
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["affine.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.cast %0 : !transform.any_op to !transform.op<"affine.for">
  %2 = transform.loop.coalesce %1 : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">)
}

// -----

func.func @coalesce_and_unroll(%arg1: memref<64x64xf32, 1>, %arg2: memref<64x64xf32, 1>, %arg3: memref<64x64xf32, 1>) attributes {} {
  // CHECK: scf.for %[[IV1:.+]] =
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  scf.for %arg4 = %c0 to %c64 step %c1 {
    // CHECK-NOT: scf.for
    scf.for %arg5 = %c0 to %c64 step %c1 {
      // CHECK: %[[IDX0:.+]] = arith.remsi %[[IV1]]
      // CHECK: %[[IDX1:.+]] = arith.divsi %[[IV1]]
      // CHECK-NEXT: %{{.+}} = memref.load %{{.+}}[%[[IDX1]], %[[IDX0]]] : memref<64x64xf32, 1>
      %0 = memref.load %arg1[%arg4, %arg5] : memref<64x64xf32, 1>
      %1 = memref.load %arg2[%arg4, %arg5] : memref<64x64xf32, 1>
      %2 = arith.addf %0, %1 : f32
      // CHECK: memref.store
      // CHECK: memref.store
      // CHECK: memref.store
      // Residual loop must have a single store.
      // CHECK: memref.store
      memref.store %2, %arg3[%arg4, %arg5] : memref<64x64xf32, 1>
    }
  } {coalesce}
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["scf.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.for">
  %2 = transform.loop.coalesce %1 : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)
  transform.loop.unroll %2 {factor = 3} : !transform.op<"scf.for">
}
