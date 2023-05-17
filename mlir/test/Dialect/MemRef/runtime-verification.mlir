// RUN: mlir-opt %s -generate-runtime-verification -cse | FileCheck %s

// CHECK-LABEL: func @expand_shape(
//  CHECK-SAME:     %[[m:.*]]: memref<?xf32>
//   CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[c5:.*]] = arith.constant 5 : index
//   CHECK-DAG:   %[[dim:.*]] = memref.dim %[[m]], %[[c0]]
//       CHECK:   %[[mod:.*]] = arith.remsi %[[dim]], %[[c5]]
//       CHECK:   %[[cmpi:.*]] = arith.cmpi eq, %[[mod]], %[[c0]]
//       CHECK:   cf.assert %[[cmpi]], "ERROR: Runtime op verification failed
func.func @expand_shape(%m: memref<?xf32>) -> memref<?x5xf32> {
  %0 = memref.expand_shape %m [[0, 1]] : memref<?xf32> into memref<?x5xf32>
  return %0 : memref<?x5xf32>
}
