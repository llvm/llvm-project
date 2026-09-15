// RUN: mlir-opt %s -generate-runtime-verification -cse -split-input-file | FileCheck %s

// CHECK-LABEL: func @expand_shape(
//  CHECK-SAME:     %[[m:.*]]: memref<?xf32>
//  CHECK-SAME:     %[[sz0:.*]]: index
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = memref.dim %[[m]], %[[c0]]
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   %[[prod:.*]] = arith.muli %[[sz0]], %[[c5]]
//       CHECK:   %[[cmpi:.*]] = arith.cmpi eq, %[[prod]], %[[dim]]
//       CHECK:   cf.assert %[[cmpi]], "ERROR: Runtime op verification failed
func.func @expand_shape(%m: memref<?xf32>, %sz0: index) -> memref<?x5xf32> {
  %0 = memref.expand_shape %m [[0, 1]] output_shape [%sz0, 5] : memref<?xf32> into memref<?x5xf32>
  return %0 : memref<?x5xf32>
}

// -----

// Ensure the SCF dialect is loaded.

// CHECK-LABEL: func @subview(
// CHECK:       scf.if
func.func @subview(%memref: memref<1xf32>, %offset: index) {
  memref.subview %memref[%offset] [1] [1] : 
      memref<1xf32> to 
      memref<1xf32, strided<[1], offset: ?>>
  return
}
