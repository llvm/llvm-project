// RUN: mlir-opt %s -linalg-fusion | FileCheck %s

func @f1(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>, %B: memref<?x?xf32, offset: ?, strides: [?, 1]>, %C: memref<?x?xf32, offset: ?, strides: [?, 1]>, %D: memref<?x?xf32, offset: ?, strides: [?, 1]>, %E: memref<?x?xf32, offset: ?, strides: [?, 1]>) -> memref<?x?xf32, offset: ?, strides: [?, 1]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %c40 = constant 40 : index
  %c30 = constant 30 : index
  %c20 = constant 20 : index
  %0 = dim %C, %c0 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  %1 = dim %C, %c1 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  %2 = dim %D, %c1 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  linalg.matmul %A, %B, %C : (memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>)
  scf.for %arg5 = %c0 to %0 step %c20 {
    scf.for %arg6 = %c0 to %2 step %c30 {
      scf.for %arg7 = %c0 to %1 step %c40 {
        %5 = std.subview %C[%arg5, %arg7][%c20, %c40][%c1, %c1] : memref<?x?xf32, offset: ?, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = std.subview %D[%arg7, %arg6][%c40, %c30][%c1, %c1]: memref<?x?xf32, offset: ?, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %E[%arg5, %arg6][%c20, %c40][%c1, %c1] : memref<?x?xf32, offset: ?, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %9 = dim %5, %c0 : memref<?x?xf32, offset: ?, strides: [?, ?]>
        %10 = dim %5, %c1 : memref<?x?xf32, offset: ?, strides: [?, ?]>
        %11 = dim %7, %c1 : memref<?x?xf32, offset: ?, strides: [?, ?]>
        scf.for %arg8 = %c0 to %9 step %c2 {
          scf.for %arg9 = %c0 to %11 step %c3 {
            scf.for %arg10 = %c0 to %10 step %c4 {
              %14 = std.subview %5[%arg8, %arg10][%c2, %c4][%c1, %c1] : memref<?x?xf32, offset: ?, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
              %16 = std.subview %7[%arg10, %arg9][%c4, %c3][%c1, %c1]: memref<?x?xf32, offset: ?, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
              %17 = std.subview %8[%arg8, %arg9][%c2, %c4][%c1, %c1] : memref<?x?xf32, offset: ?, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
              linalg.matmul %14, %16, %17 : (memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>)
            }
          }
        }
      }
    }
  }
  return %E : memref<?x?xf32, offset: ?, strides: [?, 1]>
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
