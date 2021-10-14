// RUN: mlir-opt %s -convert-linalg-to-affine-loops | FileCheck %s

// Test that we can lower all the way to LLVM without crashing, don't check results here.
// RUN: mlir-opt %s -convert-linalg-to-affine-loops -convert-linalg-to-llvm -o=/dev/null 2>&1

func @matmul(%arg0: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %A = memref.view %arg0[%c0][%M, %K] : memref<?xi8> to memref<?x?xf32>
  %B = memref.view %arg0[%c0][%K, %N] : memref<?xi8> to memref<?x?xf32>
  %C = memref.view %arg0[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32>
  linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
               outs(%C: memref<?x?xf32>)
  return
}

//----------------------------------------------------------------------------//
// Named ops to loops.
//----------------------------------------------------------------------------//
func @named_batch_matmul(%A: memref<?x?x?xf32>, %B: memref<?x?x?xf32>, %C: memref<?x?x?xf32>) {
  linalg.batch_matmul ins(%A, %B: memref<?x?x?xf32>, memref<?x?x?xf32>)
                     outs(%C : memref<?x?x?xf32>)
  return
}
// CHECK-LABEL: @named_batch_matmul
//  CHECK-SAME: %[[mA:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECK-SAME: %[[mB:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECK-SAME: %[[mC:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//       CHECK: %[[B:.*]] = memref.dim %[[mA]], %c0 : memref<?x?x?xf32>
//       CHECK: %[[M:.*]] = memref.dim %[[mA]], %c1 : memref<?x?x?xf32>
//       CHECK: %[[K:.*]] = memref.dim %[[mA]], %c2 : memref<?x?x?xf32>
//       CHECK: %[[N:.*]] = memref.dim %[[mB]], %c2 : memref<?x?x?xf32>
//       CHECK: affine.for %[[b:.*]] = {{.*}}0 to %[[B]] {
//       CHECK:   affine.for %[[m:.*]] = {{.*}}0 to %[[M]] {
//       CHECK:     affine.for %[[n:.*]] = {{.*}}0 to %[[N]] {
//       CHECK:       affine.for %[[k:.*]] = {{.*}}0 to %[[K]] {
//       CHECK:       %[[va:.*]] = affine.load %[[mA]][%[[b]], %[[m]], %[[k]]] : memref<?x?x?xf32>
//       CHECK:       %[[vb:.*]] = affine.load %[[mB]][%[[b]], %[[k]], %[[n]]] : memref<?x?x?xf32>
//       CHECK:       %[[vc:.*]] = affine.load %[[mC]][%[[b]], %[[m]], %[[n]]] : memref<?x?x?xf32>
//       CHECK:       %[[inc:.*]] = arith.mulf %[[va]], %[[vb]] : f32
//       CHECK:       %[[res:.*]] = arith.addf %[[vc]], %[[inc]] : f32
//       CHECK:       affine.store %[[res]], %[[mC]][%[[b]], %[[m]], %[[n]]] : memref<?x?x?xf32>
