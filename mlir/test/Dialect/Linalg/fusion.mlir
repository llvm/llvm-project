// RUN: mlir-opt %s -linalg-fusion -split-input-file | FileCheck %s

func @f1(%A: memref<?x?xf32, offset: 0, strides: [?, 1]>,
         %B: memref<?x?xf32, offset: 0, strides: [?, 1]>,
         %C: memref<?x?xf32, offset: 0, strides: [?, 1]>,
         %D: memref<?x?xf32, offset: 0, strides: [?, 1]>,
         %E: memref<?x?xf32, offset: 0, strides: [?, 1]>
        ) -> memref<?x?xf32, offset: 0, strides: [?, 1]> {
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %A, 0 : memref<?x?xf32, offset: 0, strides: [?, 1]>
  %1 = dim %A, 1 : memref<?x?xf32, offset: 0, strides: [?, 1]>
  %2 = dim %B, 1 : memref<?x?xf32, offset: 0, strides: [?, 1]>
  linalg.matmul(%A, %B, %C) :
    memref<?x?xf32, offset: 0, strides: [?, 1]>,
    memref<?x?xf32, offset: 0, strides: [?, 1]>,
    memref<?x?xf32, offset: 0, strides: [?, 1]>
  %c1 = constant 1 : index
  scf.for %arg5 = %c0 to %0 step %c2 {
    scf.for %arg6 = %c0 to %2 step %c3 {
      scf.for %arg7 = %c0 to %1 step %c4 {
        %5 = std.subview %A[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, 1]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = std.subview %B[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, 1]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %C[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, 1]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%5, %7, %8) :
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, 1]>
}
// CHECK-LABEL: func @f1
// CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       linalg.matmul
// CHECK:       linalg.matmul

// -----

// CHECK-DAG: #[[strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>
func @f2(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %B: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %C: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %D: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %E: memref<?x?xf32, offset: 0, strides: [?, ?]>
        ) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.matmul(%A, %B, %C) :
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>
  %0 = dim %C, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %C, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = dim %D, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  scf.for %arg5 = %c0 to %0 step %c2 {
    scf.for %arg6 = %c0 to %2 step %c3 {
      scf.for %arg7 = %c0 to %1 step %c4 {
        %5 = std.subview %C[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = std.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%5, %7, %8) :
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f2
// CHECK:  (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// CHECK-DAG:  %[[C_0:.*]] = dim %[[C]], 0 : memref<?x?xf32, #[[strided2D]]>
// CHECK-DAG:  %[[C_1:.*]] = dim %[[C]], 1 : memref<?x?xf32, #[[strided2D]]>
// CHECK-DAG:  %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
// CHECK:  scf.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
// CHECK:    scf.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
// CHECK:      scf.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
// CHECK:        linalg.matmul
// CHECK:        linalg.matmul

// -----

func @f3(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %B: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %C: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %D: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %E: memref<?x?xf32, offset: 0, strides: [?, ?]>
        ) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.matmul(%A, %B, %C) :
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>
  %0 = dim %D, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %D, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = dim %C, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  scf.for %arg5 = %c0 to %0 step %c2 {
    scf.for %arg6 = %c0 to %2 step %c3 {
      scf.for %arg7 = %c0 to %1 step %c4 {
        %5 = std.subview %D[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = std.subview %C[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%5, %7, %8) :
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f3
// CHECK:  (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// CHECK:  %[[D_0:.*]] = dim %[[D]], 0 : memref<?x?xf32, #[[strided2D]]>
// CHECK:  %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
// CHECK:  %[[C_1:.*]] = dim %[[C]], 1 : memref<?x?xf32, #[[strided2D]]>
// CHECK:  scf.for %{{.*}} = %{{.*}} to %[[D_0]] step %{{.*}} {
// CHECK:    scf.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
// CHECK:      scf.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
// CHECK:        linalg.matmul
// CHECK:        linalg.matmul

// -----

func @f4(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %B: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %C: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %D: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %E: memref<?x?xf32, offset: 0, strides: [?, ?]>
        ) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.matmul(%A, %B, %C) :
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %B, %D) :
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>
  %0 = dim %C, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %C, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = dim %D, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  scf.for %arg5 = %c0 to %0 step %c2 {
    scf.for %arg6 = %c0 to %2 step %c3 {
      scf.for %arg7 = %c0 to %1 step %c4 {
        %5 = std.subview %C[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = std.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%5, %7, %8) :
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f4
// CHECK:  (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// CHECK:  %[[C_0:.*]] = dim %[[C]], 0 : memref<?x?xf32, #[[strided2D]]>
// CHECK:  %[[C_1:.*]] = dim %[[C]], 1 : memref<?x?xf32, #[[strided2D]]>
// CHECK:  %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
// CHECK:  scf.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
// CHECK:    scf.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
// CHECK:      scf.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
// Fuse D then fuse C, no false dependence prevent it.
// CHECK:        linalg.matmul
// CHECK:        linalg.matmul
// CHECK:        linalg.matmul

// -----

// CHECK-DAG: #[[strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>
func @f5(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %B: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %C: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %D: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %E: memref<?x?xf32, offset: 0, strides: [?, ?]>
        ) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %B, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %D, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = dim %D, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %B, %C) :
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%C, %B, %D) :
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>
  scf.for %arg5 = %c0 to %1 step %c2 {
    scf.for %arg6 = %c0 to %0 step %c3 {
      scf.for %arg7 = %c0 to %2 step %c4 {
        %5 = std.subview %D[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = std.subview %B[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%5, %7, %8) :
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f5
// CHECK:  (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// CHECK-DAG:  %[[B_1:.*]] = dim %[[B]], 1 : memref<?x?xf32, #[[strided2D]]>
// CHECK-DAG:  %[[D_0:.*]] = dim %[[D]], 0 : memref<?x?xf32, #[[strided2D]]>
// CHECK-DAG:  %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
// CHECK:  scf.for %[[I:.*]] = %{{.*}} to %[[D_0]] step %{{.*}} {
// CHECK:    scf.for %[[J:.*]] = %{{.*}} to %[[B_1]] step %{{.*}} {
// CHECK:      scf.for %[[K:.*]] = %{{.*}} to %[[D_1]] step %{{.*}} {
// CHECK-DAG:    %[[D_IK:.*]] = subview %[[D]][%[[I]], %[[K]]]
// CHECK-DAG:    %[[B_KJ:.*]] = subview %[[B]][%[[K]], %[[J]]]
// CHECK-DAG:    %[[E_IJ:.*]] = subview %[[E]][%[[I]], %[[J]]]
// CHECK:        dim
// CHECK-DAG:    %[[C_I0:.*]] = subview %[[C]][%[[I]], %{{.*}}]
// CHECK-DAG:    %[[B_0K:.*]] = subview %[[B]][%{{.*}}, %[[K]]]
// CHECK-DAG:    %[[D_IK_:.*]] = subview %[[D]][%[[I]], %[[K]]]
// CHECK:        dim
// CHECK-DAG:    %[[A_I0:.*]] = subview %[[A]][%[[I]], %{{.*}}]
// CHECK-DAG:    %[[B_00:.*]] = subview %[[B]][%{{.*}}, %{{.*}}]
// CHECK-DAG:    %[[C_I0_:.*]] = subview %[[C]][%[[I]], %{{.*}}]
// CHECK:        linalg.matmul(%[[A_I0]], %[[B_00]], %[[C_I0_]])
// CHECK:        linalg.matmul(%[[C_I0]], %[[B_0K]], %[[D_IK_]])
// CHECK:        linalg.matmul(%[[D_IK]], %[[B_KJ]], %[[E_IJ]])

// -----

#map0 = affine_map<(d0) -> (d0 + 2)>
#map1 = affine_map<(d0) -> (d0 + 4)>
#map2 = affine_map<(d0) -> (d0 + 3)>

func @f6(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %B: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %C: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %D: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %E: memref<?x?xf32, offset: 0, strides: [?, ?]>
        ) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %C, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %B, %C) :
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %C, %E) :
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %C, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = dim %D, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  scf.for %arg5 = %c0 to %1 step %c2 {
    scf.for %arg6 = %c0 to %2 step %c3 {
      scf.for %arg7 = %c0 to %0 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = std.subview %C[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %6 = affine.apply #map2(%arg6)
        %7 = std.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%5, %7, %8) :
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f6
// CHECK:  (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// Fuse the producer of E (WAW) then the producer of C (WAR).
// CHECK:  scf.for
// CHECK:    scf.for
// CHECK:      scf.for
// CHECK:        linalg.matmul
// CHECK:        linalg.matmul
// CHECK:        linalg.matmul

// -----

func @f7(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %B: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %C: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %D: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %E: memref<?x?xf32, offset: 0, strides: [?, ?]>
        ) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %A, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %A, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = dim %C, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %3 = dim %C, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %4 = dim %D, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %C, %E) :
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %B, %C) :
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>
  scf.for %arg5 = %c0 to %0 step %c2 {
    scf.for %arg6 = %c0 to %2 step %c3 {
      scf.for %arg7 = %c0 to %1 step %c4 {
        %7 = std.subview %A[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %9 = std.subview %C[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %10 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%7, %9, %10) :
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  scf.for %arg5 = %c0 to %3 step %c2 {
    scf.for %arg6 = %c0 to %4 step %c3 {
      scf.for %arg7 = %c0 to %2 step %c4 {
        %7 = std.subview %C[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %9 = std.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %10 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%7, %9, %10) :
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f7
// CHECK:  (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// CHECK:  %[[A_0:.*]] = dim %[[A]], 0 : memref<?x?xf32, #[[strided2D]]>
// CHECK:  %[[A_1:.*]] = dim %[[A]], 1 : memref<?x?xf32, #[[strided2D]]>
// CHECK:  %[[C_1:.*]] = dim %[[C]], 1 : memref<?x?xf32, #[[strided2D]]>
// CHECK:  %[[C_0:.*]] = dim %[[C]], 0 : memref<?x?xf32, #[[strided2D]]>
// CHECK:  %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
// CHECK:  linalg.matmul(%[[A]], %[[C]], %[[E]])
// CHECK:  scf.for %{{.*}} = %{{.*}} to %[[A_0]] step %{{.*}} {
// CHECK:    scf.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
// CHECK:      scf.for %{{.*}} = %{{.*}} to %[[A_1]] step %{{.*}} {
// CHECK:        linalg.matmul
// CHECK:        linalg.matmul
// CHECK:  scf.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
// CHECK:    scf.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
// CHECK:      scf.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
// CHECK:        linalg.matmul
// CHECK-NOT:      linalg.matmul

// -----

#map0 = affine_map<(d0) -> (d0 + 2)>
#map1 = affine_map<(d0) -> (d0 + 4)>
#map2 = affine_map<(d0) -> (d0 + 3)>

func @f8(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %B: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %C: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %D: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %E: memref<?x?xf32, offset: 0, strides: [?, ?]>
        ) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %A, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %A, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %C, %D) :
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %B, %C) :
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = dim %D, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  scf.for %arg5 = %c0 to %0 step %c2 {
    scf.for %arg6 = %c0 to %2 step %c3 {
      scf.for %arg7 = %c0 to %1 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = std.subview %A[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %6 = affine.apply #map2(%arg6)
        %7 = std.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%5, %7, %8) :
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f8
// CHECK:  (%[[A:.*]]: memref{{.*}}, %[[B:.*]]: memref{{.*}}, %[[C:.*]]: memref{{.*}}, %[[D:.*]]: memref{{.*}}, %[[E:.*]]: memref{{.*}})
// CHECK:  linalg.matmul
// CHECK:  linalg.matmul
// CHECK:  scf.for
// CHECK:    scf.for
// CHECK:      scf.for
// CHECK:        linalg.matmul
// CHECK-NOT:      linalg.matmul

// -----

#id_2d = affine_map<(i, j) -> (i, j)>
#pointwise_2d_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  iterator_types = ["parallel", "parallel"]
}
func @pointwise(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>,
                %B: memref<?x?xf32, offset: 0, strides: [?, ?]>,
                %C: memref<?x?xf32, offset: 0, strides: [?, ?]>,
                %D: memref<?x?xf32, offset: 0, strides: [?, ?]>) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.generic #pointwise_2d_trait %A, %A, %B {
  ^bb0(%E: f32, %arg5: f32, %arg6: f32):   // no predecessors
    %2 = addf %E, %arg5 : f32
    linalg.yield %2 : f32
  }: memref<?x?xf32, offset: 0, strides: [?, ?]>,
     memref<?x?xf32, offset: 0, strides: [?, ?]>,
     memref<?x?xf32, offset: 0, strides: [?, ?]>
  %0 = dim %B, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %B, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  scf.for %arg4 = %c0 to %0 step %c2 {
    scf.for %arg5 = %c0 to %1 step %c3 {
      %4 = std.subview %B[%arg4, %arg5][%c2, %c3][%c1, %c1] :
        memref<?x?xf32, offset: 0, strides: [?, ?]> to
        memref<?x?xf32, offset: ?, strides: [?, ?]>
      %5 = std.subview %C[%arg4, %arg5][%c2, %c3][%c1, %c1] :
        memref<?x?xf32, offset: 0, strides: [?, ?]> to
        memref<?x?xf32, offset: ?, strides: [?, ?]>
      %6 = std.subview %D[%arg4, %arg5][%c2, %c3][%c1, %c1] :
        memref<?x?xf32, offset: 0, strides: [?, ?]> to
        memref<?x?xf32, offset: ?, strides: [?, ?]>
      linalg.generic #pointwise_2d_trait %4, %5, %6 {
      ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):       // no predecessors
        %7 = mulf %arg6, %arg7 : f32
        linalg.yield %7 : f32
      }: memref<?x?xf32, offset: ?, strides: [?, ?]>,
         memref<?x?xf32, offset: ?, strides: [?, ?]>,
         memref<?x?xf32, offset: ?, strides: [?, ?]>
    }
  }
  return
}
// CHECK-LABEL: func @pointwise
// CHECK:  scf.for
// CHECK:    scf.for
// CHECK-NOT:  scf.for
// CHECK:      linalg.generic
// CHECK:        addf
// CHECK:      linalg.generic
// CHECK:        mulf

// -----

#id_2d = affine_map<(i, j) -> (i, j)>
#pointwise_2d_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  iterator_types = ["parallel", "parallel"]
}
func @pointwise_no_view(%M: index, %N: index) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %A = alloc (%M, %N): memref<?x?xf32>
  %B = alloc (%M, %N): memref<?x?xf32>
  %C = alloc (%M, %N): memref<?x?xf32>
  %D = alloc (%M, %N): memref<?x?xf32>
  %E = alloc (%M, %N): memref<?x?xf32>
  linalg.generic #pointwise_2d_trait %A, %A, %B {
  ^bb0(%e: f32, %arg5: f32, %arg6: f32):   // no predecessors
    %2 = addf %e, %arg5 : f32
    linalg.yield %2 : f32
  }: memref<?x?xf32>,
     memref<?x?xf32>,
     memref<?x?xf32>
  %0 = dim %B, 0 : memref<?x?xf32>
  %1 = dim %B, 1 : memref<?x?xf32>
  scf.for %arg4 = %c0 to %0 step %c2 {
    scf.for %arg5 = %c0 to %1 step %c3 {
      %4 = std.subview %B[%arg4, %arg5][%c2, %c3][%c1, %c1] :
        memref<?x?xf32> to
        memref<?x?xf32, offset: ?, strides: [?, ?]>
      %5 = std.subview %C[%arg4, %arg5][%c2, %c3][%c1, %c1] :
        memref<?x?xf32> to
        memref<?x?xf32, offset: ?, strides: [?, ?]>
      %6 = std.subview %D[%arg4, %arg5][%c2, %c3][%c1, %c1] :
        memref<?x?xf32> to
        memref<?x?xf32, offset: ?, strides: [?, ?]>
      linalg.generic #pointwise_2d_trait %4, %5, %6 {
      ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):       // no predecessors
        %7 = mulf %arg6, %arg7 : f32
        linalg.yield %7 : f32
      }: memref<?x?xf32, offset: ?, strides: [?, ?]>,
         memref<?x?xf32, offset: ?, strides: [?, ?]>,
         memref<?x?xf32, offset: ?, strides: [?, ?]>
    }
  }
  return
}
// CHECK-LABEL: func @pointwise_no_view
// CHECK:  scf.for
// CHECK:    scf.for
// CHECK-NOT:  scf.for
// CHECK:      linalg.generic
// CHECK:        addf
// CHECK:      linalg.generic
// CHECK:        mulf


// -----

#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>

func @fusion_of_three(%arg0: memref<100x10xf32>,
                      %arg1: memref<100xf32>,
                      %arg2: memref<100x10xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = alloc() {temp = true} : memref<100x10xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "parallel"]
  } %arg1, %0 {
      ^bb0(%arg3: f32, %arg4: f32): // no predecessors
        linalg.yield %arg3 : f32
      }: memref<100xf32>, memref<100x10xf32>
  %1 = alloc() {temp = true} : memref<100x10xf32>
  linalg.generic {
    args_in = 2 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map1, #map1, #map1],
    iterator_types = ["parallel", "parallel"]
  } %arg0, %0, %1 {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32): // no predecessors
        %2 = subf %arg3, %arg4 : f32
        linalg.yield %2 : f32
      }: memref<100x10xf32>, memref<100x10xf32>, memref<100x10xf32>
  dealloc %0 : memref<100x10xf32>
  %2 = dim %1, 0 : memref<100x10xf32>
  %3 = dim %1, 1 : memref<100x10xf32>
  %4 = dim %arg2, 0 : memref<100x10xf32>
  %5 = dim %arg2, 1 : memref<100x10xf32>
  scf.for %i = %c0 to %2 step %c1 {
    scf.for %j = %c0 to %3 step %c1 {
      %6 = std.subview %1[%i, %j][%c1, %c1][%c1, %c1] :
      memref<100x10xf32> to memref<?x?xf32, #map2>
      %7 = std.subview %arg2[%i, %j][%c1, %c1][%c1, %c1] :
      memref<100x10xf32> to memref<?x?xf32, #map2>
      linalg.generic {
        args_in = 1 : i64,
        args_out = 1 : i64,
        indexing_maps = [#map1, #map1],
        iterator_types = ["parallel", "parallel"]
      } %6, %7 {
          ^bb0(%arg3: f32, %arg4: f32):     // no predecessors
            %8 = exp %arg3 : f32
            linalg.yield %8 : f32
          }: memref<?x?xf32, #map2>,
             memref<?x?xf32, #map2>
    }
  }
 dealloc %1 : memref<100x10xf32>
 return
}
// CHECK-LABEL: func @fusion
// CHECK-NOT: linalg.generic
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK-NOT: scf.for
// CHECK:       linalg.generic
// CHECK:         linalg.yield
// CHECK:       linalg.generic
// CHECK:         subf
// CHECK:         linalg.yield
// CHECK:       linalg.generic
// CHECK:         exp
// CHECK:         linalg.yield

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1 - d2)>
#map1 = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4)>
#map2 = affine_map<()[s0] -> (s0 + 3)>

func @fill_and_conv(%arg0: memref<1x4x5x1xf32>, %arg1: memref<2x3x1x1xf32>, %arg2: memref<1x4x5x1xf32>) {
  %cst = constant 0.000000e+00 : f32
  linalg.fill(%arg2, %cst) : memref<1x4x5x1xf32>, f32

  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %4 = dim %arg1, 0 : memref<2x3x1x1xf32>
  %5 = dim %arg1, 1 : memref<2x3x1x1xf32>
  %6 = dim %arg0, 0 : memref<1x4x5x1xf32>
  %7 = dim %arg0, 1 : memref<1x4x5x1xf32>
  %8 = dim %arg0, 3 : memref<1x4x5x1xf32>
  %9 = dim %arg2, 0 : memref<1x4x5x1xf32>
  %10 = dim %arg2, 1 : memref<1x4x5x1xf32>
  %11 = dim %arg2, 2 : memref<1x4x5x1xf32>
  %12 = dim %arg2, 3 : memref<1x4x5x1xf32>
  %13 = linalg.range %c0 : %6 : %c2 : !linalg.range
  %14 = linalg.range %c0 : %10 : %c3 : !linalg.range
  scf.for %arg3 = %c0 to %6 step %c2 {
    scf.for %arg4 = %c0 to %10 step %c3 {
      %15 = affine.min #map0(%c2, %c1, %arg3)
      %16 = affine.apply #map2()[%7]
      %17 = affine.min #map0(%16, %c4, %arg4)
      %18 = dim %arg0, 2 : memref<1x4x5x1xf32>
      %19 = dim %arg0, 3 : memref<1x4x5x1xf32>
      %20 = subview %arg0[%arg3, %arg4, %c0, %c0] [%15, %17, %18, %19] [%c1, %c1, %c1, %c1] : memref<1x4x5x1xf32> to memref<?x?x?x?xf32, #map1>
      %21 = affine.min #map0(%c2, %c1, %arg3)
      %22 = affine.min #map0(%c3, %c4, %arg4)
      %23 = dim %arg2, 2 : memref<1x4x5x1xf32>
      %24 = dim %arg2, 3 : memref<1x4x5x1xf32>
      %25 = subview %arg2[%arg3, %arg4, %c0, %c0] [%21, %22, %23, %24] [%c1, %c1, %c1, %c1] : memref<1x4x5x1xf32> to memref<?x?x?x?xf32, #map1>
      linalg.conv(%arg1, %20, %25) {dilations = [1, 1], strides = [1, 1]} : memref<2x3x1x1xf32>, memref<?x?x?x?xf32, #map1>, memref<?x?x?x?xf32, #map1>
    }
  }
  return
}
// CHECK-LABEL: func @fill_and_conv
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.fill
// CHECK:     linalg.conv

// -----

// Test that different allocation-like ops are recognized and properly handled.
func @accept_different_alloc_ops(%dim: index, %s0 : index, %s1: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index

  %A = alloca(%dim, %dim)[%s0, %s1] : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %B = alloca(%dim, %dim)[%s0, %s1] : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %C = alloc(%dim, %dim)[%s0, %s1]  : memref<?x?xf32, offset: 0, strides: [?, ?]>

  linalg.matmul(%A, %B, %C) :
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>,
    memref<?x?xf32, offset: 0, strides: [?, ?]>

  scf.for %i = %c0 to %dim step %c2 {
    scf.for %j = %c0 to %dim step %c3 {
      scf.for %k = %c0 to %dim step %c4 {
        %0 = std.subview %A[%i, %k][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %1 = std.subview %B[%k, %j][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %2 = std.subview %C[%i, %j][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%0, %1, %2) :
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>,
          memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return
}

// CHECK-LABEL: func @accept_different_alloc_ops
// CHECK-COUNT-3: scf.for
// CHECK-COUNT-2:   linalg.matmul
