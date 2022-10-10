// RUN: mlir-opt %s -test-linalg-greedy-fusion | FileCheck %s

func.func @f1(%A: memref<?x?xf32, strided<[?, 1], offset: ?>>, %B: memref<?x?xf32, strided<[?, 1], offset: ?>>, %C: memref<?x?xf32, strided<[?, 1], offset: ?>>, %D: memref<?x?xf32, strided<[?, 1], offset: ?>>, %E: memref<?x?xf32, strided<[?, 1], offset: ?>>) -> memref<?x?xf32, strided<[?, 1], offset: ?>> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c40 = arith.constant 40 : index
  %c30 = arith.constant 30 : index
  %c20 = arith.constant 20 : index
  %0 = memref.dim %C, %c0 : memref<?x?xf32, strided<[?, 1], offset: ?>>
  %1 = memref.dim %C, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
  %2 = memref.dim %D, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
  linalg.matmul ins(%A, %B: memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>)
               outs(%C: memref<?x?xf32, strided<[?, 1], offset: ?>>)
  scf.for %arg5 = %c0 to %0 step %c20 {
    scf.for %arg6 = %c0 to %2 step %c30 {
      scf.for %arg7 = %c0 to %1 step %c40 {
        %5 = memref.subview %C[%arg5, %arg7][%c20, %c40][%c1, %c1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %7 = memref.subview %D[%arg7, %arg6][%c40, %c30][%c1, %c1]: memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %8 = memref.subview %E[%arg5, %arg6][%c20, %c40][%c1, %c1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %9 = memref.dim %5, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %10 = memref.dim %5, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %11 = memref.dim %7, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
        scf.for %arg8 = %c0 to %9 step %c2 {
          scf.for %arg9 = %c0 to %11 step %c3 {
            scf.for %arg10 = %c0 to %10 step %c4 {
              %14 = memref.subview %5[%arg8, %arg10][%c2, %c4][%c1, %c1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %16 = memref.subview %7[%arg10, %arg9][%c4, %c3][%c1, %c1]: memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %17 = memref.subview %8[%arg8, %arg9][%c2, %c3][%c1, %c1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              linalg.matmul ins(%14, %16: memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>)
                           outs(%17: memref<?x?xf32, strided<[?, ?], offset: ?>>)
            }
          }
        }
      }
    }
  }
  return %E : memref<?x?xf32, strided<[?, 1], offset: ?>>
}
// CHECK-LABEL: func @f1
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//      CHECK: scf.for
//      CHECK:   scf.for
//      CHECK:     scf.for
//      CHECK:      scf.for
//      CHECK:        scf.for
//      CHECK:          scf.for
//      CHECK:            linalg.matmul
//      CHECK:            linalg.matmul
