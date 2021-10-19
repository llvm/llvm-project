// RUN: mlir-opt %s -test-linalg-greedy-fusion -split-input-file | FileCheck %s

func @f1(%A: memref<?x?xf32, offset: 0, strides: [?, 1]>,
         %B: memref<?x?xf32, offset: 0, strides: [?, 1]>,
         %C: memref<?x?xf32, offset: 0, strides: [?, 1]>,
         %D: memref<?x?xf32, offset: 0, strides: [?, 1]>,
         %E: memref<?x?xf32, offset: 0, strides: [?, 1]>
        ) -> memref<?x?xf32, offset: 0, strides: [?, 1]> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %A, %c0 : memref<?x?xf32, offset: 0, strides: [?, 1]>
  %1 = memref.dim %A, %c1 : memref<?x?xf32, offset: 0, strides: [?, 1]>
  %2 = memref.dim %B, %c1 : memref<?x?xf32, offset: 0, strides: [?, 1]>
  linalg.matmul ins(%A, %B : memref<?x?xf32, offset: 0, strides: [?, 1]>,
                             memref<?x?xf32, offset: 0, strides: [?, 1]>)
               outs(%C : memref<?x?xf32, offset: 0, strides: [?, 1]>)
  scf.for %arg5 = %c0 to %0 step %c2 {
    scf.for %arg6 = %c0 to %2 step %c3 {
      scf.for %arg7 = %c0 to %1 step %c4 {
        %5 = memref.subview %A[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, 1]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = memref.subview %B[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, 1]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = memref.subview %C[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, 1]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul ins(%5, %7 : memref<?x?xf32, offset: ?, strides: [?, ?]>,
                                   memref<?x?xf32, offset: ?, strides: [?, ?]>)
                     outs(%8: memref<?x?xf32, offset: ?, strides: [?, ?]>)
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

// CHECK-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>
func @f2(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %B: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %C: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %D: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %E: memref<?x?xf32, offset: 0, strides: [?, ?]>
        ) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  linalg.matmul ins(%A, %B : memref<?x?xf32, offset: 0, strides: [?, ?]>,
                             memref<?x?xf32, offset: 0, strides: [?, ?]>)
               outs(%C: memref<?x?xf32, offset: 0, strides: [?, ?]>)
  %0 = memref.dim %C, %c0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = memref.dim %C, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = memref.dim %D, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  scf.for %arg5 = %c0 to %0 step %c2 {
    scf.for %arg6 = %c0 to %2 step %c3 {
      scf.for %arg7 = %c0 to %1 step %c4 {
        %5 = memref.subview %C[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = memref.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = memref.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul ins(%5, %7 : memref<?x?xf32, offset: ?, strides: [?, ?]>,
                                   memref<?x?xf32, offset: ?, strides: [?, ?]>)
                     outs(%8 : memref<?x?xf32, offset: ?, strides: [?, ?]>)
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f2
// CHECK:  (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// CHECK-DAG:  %[[C_0:.*]] = memref.dim %[[C]], %c0{{[_0-9]*}} : memref<?x?xf32, #[[$strided2D]]>
// CHECK-DAG:  %[[C_1:.*]] = memref.dim %[[C]], %c1{{[_0-9]*}} : memref<?x?xf32, #[[$strided2D]]>
// CHECK-DAG:  %[[D_1:.*]] = memref.dim %[[D]], %c1{{[_0-9]*}} : memref<?x?xf32, #[[$strided2D]]>
// CHECK:  scf.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
// CHECK:    scf.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
// CHECK:      scf.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
// CHECK:        linalg.matmul
// CHECK:        linalg.matmul

// -----

// CHECK-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>

func @f3(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %B: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %C: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %D: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %E: memref<?x?xf32, offset: 0, strides: [?, ?]>
        ) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  linalg.matmul ins(%A, %B : memref<?x?xf32, offset: 0, strides: [?, ?]>,
                             memref<?x?xf32, offset: 0, strides: [?, ?]>)
               outs(%C : memref<?x?xf32, offset: 0, strides: [?, ?]>)
  %0 = memref.dim %D, %c0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = memref.dim %D, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = memref.dim %C, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  scf.for %arg5 = %c0 to %0 step %c2 {
    scf.for %arg6 = %c0 to %2 step %c3 {
      scf.for %arg7 = %c0 to %1 step %c4 {
        %5 = memref.subview %D[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = memref.subview %C[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = memref.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul ins(%5, %7 : memref<?x?xf32, offset: ?, strides: [?, ?]>,
                                   memref<?x?xf32, offset: ?, strides: [?, ?]>)
                     outs(%8 : memref<?x?xf32, offset: ?, strides: [?, ?]>)
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f3
// CHECK:  (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK:  %[[D_0:.*]] = memref.dim %[[D]], %[[C0]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK:  %[[D_1:.*]] = memref.dim %[[D]], %[[C1]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK:  %[[C_1:.*]] = memref.dim %[[C]], %[[C1]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK:  scf.for %{{.*}} = %{{.*}} to %[[D_0]] step %{{.*}} {
// CHECK:    scf.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
// CHECK:      scf.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
// CHECK:        linalg.matmul
// CHECK:        linalg.matmul

// -----

// CHECK-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>

func @f4(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %B: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %C: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %D: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %E: memref<?x?xf32, offset: 0, strides: [?, ?]>
        ) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  linalg.matmul ins(%A, %B : memref<?x?xf32, offset: 0, strides: [?, ?]>,
                             memref<?x?xf32, offset: 0, strides: [?, ?]>)
               outs(%C : memref<?x?xf32, offset: 0, strides: [?, ?]>)
  linalg.matmul ins(%A, %B : memref<?x?xf32, offset: 0, strides: [?, ?]>,
                             memref<?x?xf32, offset: 0, strides: [?, ?]>)
               outs(%D : memref<?x?xf32, offset: 0, strides: [?, ?]>)
  %0 = memref.dim %C, %c0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = memref.dim %C, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = memref.dim %D, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  scf.for %arg5 = %c0 to %0 step %c2 {
    scf.for %arg6 = %c0 to %2 step %c3 {
      scf.for %arg7 = %c0 to %1 step %c4 {
        %5 = memref.subview %C[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = memref.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = memref.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul ins(%5, %7 : memref<?x?xf32, offset: ?, strides: [?, ?]>,
                                   memref<?x?xf32, offset: ?, strides: [?, ?]>)
                     outs(%8 : memref<?x?xf32, offset: ?, strides: [?, ?]>)
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f4
// CHECK:  (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK:  %[[C_0:.*]] = memref.dim %[[C]], %[[C0:.*]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK:  %[[C_1:.*]] = memref.dim %[[C]], %[[C1:.*]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK:  %[[D_1:.*]] = memref.dim %[[D]], %[[C1:.*]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK:  scf.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
// CHECK:    scf.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
// CHECK:      scf.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
// Fuse D then fuse C, no false dependence prevent it.
// CHECK:        linalg.matmul
// CHECK:        linalg.matmul
// CHECK:        linalg.matmul

// -----

// CHECK-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>
func @f5(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %B: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %C: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %D: memref<?x?xf32, offset: 0, strides: [?, ?]>,
         %E: memref<?x?xf32, offset: 0, strides: [?, ?]>
        ) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %0 = memref.dim %B, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = memref.dim %D, %c0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = memref.dim %D, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul ins(%A, %B : memref<?x?xf32, offset: 0, strides: [?, ?]>,
                             memref<?x?xf32, offset: 0, strides: [?, ?]>)
               outs(%C : memref<?x?xf32, offset: 0, strides: [?, ?]>)
  linalg.matmul ins(%C, %B : memref<?x?xf32, offset: 0, strides: [?, ?]>,
                             memref<?x?xf32, offset: 0, strides: [?, ?]>)
               outs(%D : memref<?x?xf32, offset: 0, strides: [?, ?]>)
  scf.for %arg5 = %c0 to %1 step %c2 {
    scf.for %arg6 = %c0 to %0 step %c3 {
      scf.for %arg7 = %c0 to %2 step %c4 {
        %5 = memref.subview %D[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = memref.subview %B[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = memref.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul ins(%5, %7 : memref<?x?xf32, offset: ?, strides: [?, ?]>,
                                   memref<?x?xf32, offset: ?, strides: [?, ?]>)
                     outs(%8 : memref<?x?xf32, offset: ?, strides: [?, ?]>)
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}

// CHECK-DAG: #[[BOUND_2_MAP:.+]] = affine_map<(d0)[s0] -> (2, -d0 + s0)>
// CHECK-DAG: #[[BOUND_2_MAP_2:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s0, 2, -d0 + s1)>
// CHECK-DAG: #[[BOUND_4_MAP:.+]] = affine_map<(d0)[s0] -> (4, -d0 + s0)>
// CHECK: func @f5
// CHECK-SAME:  (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:  %[[A_0:.*]] = memref.dim %[[A]], %[[C0]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK-DAG:  %[[B_1:.*]] = memref.dim %[[B]], %[[C1]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK-DAG:  %[[C_0:.*]] = memref.dim %[[C]], %[[C0]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK-DAG:  %[[D_0:.*]] = memref.dim %[[D]], %[[C0]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK-DAG:  %[[D_1:.*]] = memref.dim %[[D]], %[[C1]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK-DAG:  %[[B_00:.*]] = memref.subview %[[B]][0, 0]{{.*}}
//     CHECK:  scf.for %[[I:.*]] = %{{.*}} to %[[D_0]] step %{{.*}} {
//     CHECK:    %[[BOUND_2_C0:.+]] = affine.min #[[BOUND_2_MAP]](%[[I]])[%[[C_0]]]
//     CHECK:    %[[C_I0:.*]] = memref.subview %[[C]][%[[I]], 0] [%[[BOUND_2_C0]]
//     CHECK:    %[[BOUND_ID_C0:.+]] = affine.min #[[BOUND_2_MAP_2]](%[[I]])[%[[A_0]], %[[C_0]]]
//     CHECK:    %[[A_I0:.*]] = memref.subview %[[A]][%[[I]], 0]
//     CHECK:    %[[C_I0_OUT:.*]] = memref.subview %[[C]][%[[I]], 0] [%[[BOUND_ID_C0]]
//     CHECK:    scf.for %[[J:.*]] = %{{.*}} to %[[B_1]] step %{{.*}} {
//     CHECK:      %[[E_IJ:.*]] = memref.subview %[[E]][%[[I]], %[[J]]]
//     CHECK:      scf.for %[[K:.*]] = %{{.*}} to %[[D_1]] step %{{.*}} {
//     CHECK:        %[[D_IK:.*]] = memref.subview %[[D]][%[[I]], %[[K]]] [2, 4]
//     CHECK:        %[[B_KJ:.*]] = memref.subview %[[B]][%[[K]], %[[J]]]
//     CHECK:        %[[BOUND_4_B1:.*]] = affine.min #[[BOUND_4_MAP]](%[[K]])[%[[B_1]]]
//     CHECK:        %[[B_0K:.*]] = memref.subview %[[B]][0, %[[K]]]
//     CHECK:        %[[D_IK_OUT:.+]] = memref.subview %[[D]][%[[I]], %[[K]]] [%[[BOUND_2_C0]], %[[BOUND_4_B1]]]
//     CHECK:        linalg.matmul ins(%[[A_I0]], %[[B_00]]{{.*}} outs(%[[C_I0_OUT]]
//     CHECK:        linalg.matmul ins(%[[C_I0]], %[[B_0K]]{{.*}} outs(%[[D_IK_OUT]]
//     CHECK:        linalg.matmul ins(%[[D_IK]], %[[B_KJ]]{{.*}} outs(%[[E_IJ]]

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
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %0 = memref.dim %C, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul ins(%A, %B : memref<?x?xf32, offset: 0, strides: [?, ?]>,
                             memref<?x?xf32, offset: 0, strides: [?, ?]>)
               outs(%C : memref<?x?xf32, offset: 0, strides: [?, ?]>)
  linalg.matmul ins(%A, %C : memref<?x?xf32, offset: 0, strides: [?, ?]>,
                             memref<?x?xf32, offset: 0, strides: [?, ?]>)
               outs(%E : memref<?x?xf32, offset: 0, strides: [?, ?]>)
  %1 = memref.dim %C, %c0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = memref.dim %D, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  scf.for %arg5 = %c0 to %1 step %c2 {
    scf.for %arg6 = %c0 to %2 step %c3 {
      scf.for %arg7 = %c0 to %0 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = memref.subview %C[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %6 = affine.apply #map2(%arg6)
        %7 = memref.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = memref.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul ins(%5, %7 : memref<?x?xf32, offset: ?, strides: [?, ?]>,
                                   memref<?x?xf32, offset: ?, strides: [?, ?]>)
                     outs(%8 : memref<?x?xf32, offset: ?, strides: [?, ?]>)
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
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %0 = memref.dim %A, %c0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = memref.dim %A, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = memref.dim %C, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %3 = memref.dim %C, %c0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %4 = memref.dim %D, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul ins(%A, %C : memref<?x?xf32, offset: 0, strides: [?, ?]>,
                             memref<?x?xf32, offset: 0, strides: [?, ?]>)
               outs(%E : memref<?x?xf32, offset: 0, strides: [?, ?]>)
  linalg.matmul ins(%A, %B : memref<?x?xf32, offset: 0, strides: [?, ?]>,
                             memref<?x?xf32, offset: 0, strides: [?, ?]>)
               outs(%C : memref<?x?xf32, offset: 0, strides: [?, ?]>)
  scf.for %arg5 = %c0 to %0 step %c2 {
    scf.for %arg6 = %c0 to %2 step %c3 {
      scf.for %arg7 = %c0 to %1 step %c4 {
        %7 = memref.subview %A[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %9 = memref.subview %C[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %10 = memref.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul ins(%7, %9 : memref<?x?xf32, offset: ?, strides: [?, ?]>,
                                   memref<?x?xf32, offset: ?, strides: [?, ?]>)
                     outs(%10 : memref<?x?xf32, offset: ?, strides: [?, ?]>)
      }
    }
  }
  scf.for %arg5 = %c0 to %3 step %c2 {
    scf.for %arg6 = %c0 to %4 step %c3 {
      scf.for %arg7 = %c0 to %2 step %c4 {
        %7 = memref.subview %C[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %9 = memref.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %10 = memref.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul ins(%7, %9 : memref<?x?xf32, offset: ?, strides: [?, ?]>,
                                   memref<?x?xf32, offset: ?, strides: [?, ?]>)
                     outs(%10 : memref<?x?xf32, offset: ?, strides: [?, ?]>)
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f7
// CHECK:  (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK:  %[[A_0:.*]] = memref.dim %[[A]], %[[C0:.*]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK:  %[[A_1:.*]] = memref.dim %[[A]], %[[C1:.*]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK:  %[[C_1:.*]] = memref.dim %[[C]], %[[C1:.*]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK:  %[[C_0:.*]] = memref.dim %[[C]], %[[C0:.*]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK:  %[[D_1:.*]] = memref.dim %[[D]], %[[C1:.*]] : memref<?x?xf32, #[[$strided2D]]>
// CHECK:  linalg.matmul ins(%[[A]], %[[C]]{{.*}} outs(%[[E]]
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
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %0 = memref.dim %A, %c0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = memref.dim %A, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul ins(%A, %C : memref<?x?xf32, offset: 0, strides: [?, ?]>,
                             memref<?x?xf32, offset: 0, strides: [?, ?]>)
               outs(%D : memref<?x?xf32, offset: 0, strides: [?, ?]>)
  linalg.matmul ins(%A, %B : memref<?x?xf32, offset: 0, strides: [?, ?]>,
                             memref<?x?xf32, offset: 0, strides: [?, ?]>)
               outs(%C : memref<?x?xf32, offset: 0, strides: [?, ?]>)
  %2 = memref.dim %D, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  scf.for %arg5 = %c0 to %0 step %c2 {
    scf.for %arg6 = %c0 to %2 step %c3 {
      scf.for %arg7 = %c0 to %1 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = memref.subview %A[%arg5, %arg7][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %6 = affine.apply #map2(%arg6)
        %7 = memref.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = memref.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul ins(%5, %7 : memref<?x?xf32, offset: ?, strides: [?, ?]>,
                                   memref<?x?xf32, offset: ?, strides: [?, ?]>)
                     outs(%8 : memref<?x?xf32, offset: ?, strides: [?, ?]>)
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
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  iterator_types = ["parallel", "parallel"]
}
func @pointwise(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>,
                %B: memref<?x?xf32, offset: 0, strides: [?, ?]>,
                %C: memref<?x?xf32, offset: 0, strides: [?, ?]>,
                %D: memref<?x?xf32, offset: 0, strides: [?, ?]>) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  linalg.generic #pointwise_2d_trait
      ins(%A, %A: memref<?x?xf32, offset: 0, strides: [?, ?]>,
                  memref<?x?xf32, offset: 0, strides: [?, ?]>)
     outs(%B : memref<?x?xf32, offset: 0, strides: [?, ?]>) {
  ^bb0(%E: f32, %arg5: f32, %arg6: f32):   // no predecessors
    %2 = arith.addf %E, %arg5 : f32
    linalg.yield %2 : f32
  }
  %0 = memref.dim %B, %c0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = memref.dim %B, %c1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  scf.for %arg4 = %c0 to %0 step %c2 {
    scf.for %arg5 = %c0 to %1 step %c3 {
      %4 = memref.subview %B[%arg4, %arg5][%c2, %c3][%c1, %c1] :
        memref<?x?xf32, offset: 0, strides: [?, ?]> to
        memref<?x?xf32, offset: ?, strides: [?, ?]>
      %5 = memref.subview %C[%arg4, %arg5][%c2, %c3][%c1, %c1] :
        memref<?x?xf32, offset: 0, strides: [?, ?]> to
        memref<?x?xf32, offset: ?, strides: [?, ?]>
      %6 = memref.subview %D[%arg4, %arg5][%c2, %c3][%c1, %c1] :
        memref<?x?xf32, offset: 0, strides: [?, ?]> to
        memref<?x?xf32, offset: ?, strides: [?, ?]>
      linalg.generic #pointwise_2d_trait
        ins(%4, %5: memref<?x?xf32, offset: ?, strides: [?, ?]>,
                    memref<?x?xf32, offset: ?, strides: [?, ?]>)
       outs(%6 : memref<?x?xf32, offset: ?, strides: [?, ?]>) {
      ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):       // no predecessors
        %7 = arith.mulf %arg6, %arg7 : f32
        linalg.yield %7 : f32
      }
    }
  }
  return
}
// CHECK-LABEL: func @pointwise
// CHECK:  scf.for
// CHECK:    scf.for
// CHECK-NOT:  scf.for
// CHECK:      linalg.generic
// CHECK:        arith.addf
// CHECK:      linalg.generic
// CHECK:        arith.mulf

// -----

#id_2d = affine_map<(i, j) -> (i, j)>
#pointwise_2d_trait = {
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  iterator_types = ["parallel", "parallel"]
}
func @pointwise_no_view(%M: index, %N: index) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %A = memref.alloc (%M, %N): memref<?x?xf32>
  %B = memref.alloc (%M, %N): memref<?x?xf32>
  %C = memref.alloc (%M, %N): memref<?x?xf32>
  %D = memref.alloc (%M, %N): memref<?x?xf32>
  %E = memref.alloc (%M, %N): memref<?x?xf32>
  linalg.generic #pointwise_2d_trait
    ins(%A, %A : memref<?x?xf32>, memref<?x?xf32>)
   outs(%B : memref<?x?xf32>) {
  ^bb0(%e: f32, %arg5: f32, %arg6: f32):   // no predecessors
    %2 = arith.addf %e, %arg5 : f32
    linalg.yield %2 : f32
  }
  %0 = memref.dim %B, %c0 : memref<?x?xf32>
  %1 = memref.dim %B, %c1 : memref<?x?xf32>
  scf.for %arg4 = %c0 to %0 step %c2 {
    scf.for %arg5 = %c0 to %1 step %c3 {
      %4 = memref.subview %B[%arg4, %arg5][%c2, %c3][%c1, %c1] :
        memref<?x?xf32> to
        memref<?x?xf32, offset: ?, strides: [?, ?]>
      %5 = memref.subview %C[%arg4, %arg5][%c2, %c3][%c1, %c1] :
        memref<?x?xf32> to
        memref<?x?xf32, offset: ?, strides: [?, ?]>
      %6 = memref.subview %D[%arg4, %arg5][%c2, %c3][%c1, %c1] :
        memref<?x?xf32> to
        memref<?x?xf32, offset: ?, strides: [?, ?]>
      linalg.generic #pointwise_2d_trait
        ins(%4, %5: memref<?x?xf32, offset: ?, strides: [?, ?]>,
                    memref<?x?xf32, offset: ?, strides: [?, ?]>)
       outs(%6 : memref<?x?xf32, offset: ?, strides: [?, ?]>) {
      ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):       // no predecessors
        %7 = arith.mulf %arg6, %arg7 : f32
        linalg.yield %7 : f32
      }
    }
  }
  return
}
// CHECK-LABEL: func @pointwise_no_view
// CHECK:  scf.for
// CHECK:    scf.for
// CHECK-NOT:  scf.for
// CHECK:      linalg.generic
// CHECK:        arith.addf
// CHECK:      linalg.generic
// CHECK:        arith.mulf


// -----

#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>

func @fusion_of_three(%arg0: memref<100x10xf32>,
                      %arg1: memref<100xf32>,
                      %arg2: memref<100x10xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.alloc() {temp = true} : memref<100x10xf32>
  linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<100xf32>)
   outs(%0 : memref<100x10xf32>) {
      ^bb0(%arg3: f32, %arg4: f32): // no predecessors
        linalg.yield %arg3 : f32
      }
  %1 = memref.alloc() {temp = true} : memref<100x10xf32>
  linalg.generic {
    indexing_maps = [#map1, #map1, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %0: memref<100x10xf32>, memref<100x10xf32>)
   outs(%1 : memref<100x10xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32): // no predecessors
        %2 = arith.subf %arg3, %arg4 : f32
        linalg.yield %2 : f32
      }
  memref.dealloc %0 : memref<100x10xf32>
  %2 = memref.dim %1, %c0 : memref<100x10xf32>
  %3 = memref.dim %1, %c1 : memref<100x10xf32>
  %4 = memref.dim %arg2, %c0 : memref<100x10xf32>
  %5 = memref.dim %arg2, %c1 : memref<100x10xf32>
  scf.for %i = %c0 to %2 step %c1 {
    scf.for %j = %c0 to %3 step %c1 {
      %6 = memref.subview %1[%i, %j][%c1, %c1][%c1, %c1] :
      memref<100x10xf32> to memref<?x?xf32, #map2>
      %7 = memref.subview %arg2[%i, %j][%c1, %c1][%c1, %c1] :
      memref<100x10xf32> to memref<?x?xf32, #map2>
      linalg.generic {
        indexing_maps = [#map1, #map1],
        iterator_types = ["parallel", "parallel"]}
        ins(%6 : memref<?x?xf32, #map2>)
       outs(%7 : memref<?x?xf32, #map2>) {
          ^bb0(%arg3: f32, %arg4: f32):     // no predecessors
            %8 = math.exp %arg3 : f32
            linalg.yield %8 : f32
          }
    }
  }
 memref.dealloc %1 : memref<100x10xf32>
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
// CHECK:         arith.subf
// CHECK:         linalg.yield
// CHECK:       linalg.generic
// CHECK:         exp
// CHECK:         linalg.yield

// -----


#map0 = affine_map<(d0)[s0] -> (2, -d0 + s0)>
#map1 = affine_map<(d0)[s0] -> (3, -d0 + s0)>
#map2 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
#map3 = affine_map<(d0)[s0, s1] -> (s0 + 1, -d0 + s0 + s1)>
#map4 = affine_map<(d0)[s0, s1] -> (s0 + 2, -d0 + s0 + s1)>

func @fill_and_conv(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  linalg.fill(%cst, %arg0) : f32, memref<?x?xf32>
  %2 = memref.dim %arg1, %c0 : memref<?x?xf32>
  %3 = memref.dim %arg1, %c1 : memref<?x?xf32>
  %4 = memref.dim %arg2, %c0 : memref<?x?xf32>
  %5 = memref.dim %arg2, %c1 : memref<?x?xf32>
  scf.for %arg3 = %c0 to %4 step %c2 {
    scf.for %arg4 = %c0 to %5 step %c3 {
      %6 = affine.min #map3(%arg3)[%2, %4]
      %7 = affine.min #map4(%arg4)[%3, %5]
      %8 = memref.subview %arg0[%arg3, %arg4] [%6, %7] [1, 1] : memref<?x?xf32> to memref<?x?xf32, #map2>
      %9 = affine.min #map0(%arg3)[%4]
      %10 = affine.min #map1(%arg4)[%5]
      %11 = memref.subview %arg2[%arg3, %arg4] [%9, %10] [1, 1] : memref<?x?xf32> to memref<?x?xf32, #map2>
      linalg.conv_2d ins(%8, %arg1 : memref<?x?xf32, #map2>, memref<?x?xf32>) outs(%11 : memref<?x?xf32, #map2>)
    }
  }
  return
}
// CHECK-LABEL: func @fill_and_conv
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.fill
// CHECK:     linalg.conv_2d

// -----

// Test that different allocation-like ops are recognized and properly handled.
func @accept_different_alloc_ops(%dim: index, %s0 : index, %s1: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  %A = memref.alloca(%dim, %dim)[%s0, %s1] : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %B = memref.alloca(%dim, %dim)[%s0, %s1] : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %C = memref.alloc(%dim, %dim)[%s0, %s1]  : memref<?x?xf32, offset: 0, strides: [?, ?]>

  linalg.matmul ins(%A, %B : memref<?x?xf32, offset: 0, strides: [?, ?]>,
                             memref<?x?xf32, offset: 0, strides: [?, ?]>)
               outs(%C : memref<?x?xf32, offset: 0, strides: [?, ?]>)

  scf.for %i = %c0 to %dim step %c2 {
    scf.for %j = %c0 to %dim step %c3 {
      scf.for %k = %c0 to %dim step %c4 {
        %0 = memref.subview %A[%i, %k][%c2, %c4][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %1 = memref.subview %B[%k, %j][%c4, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        %2 = memref.subview %C[%i, %j][%c2, %c3][%c1, %c1] :
          memref<?x?xf32, offset: 0, strides: [?, ?]> to
          memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul ins(%0, %1 : memref<?x?xf32, offset: ?, strides: [?, ?]>,
                                   memref<?x?xf32, offset: ?, strides: [?, ?]>)
                     outs(%2 : memref<?x?xf32, offset: ?, strides: [?, ?]>)
      }
    }
  }
  return
}

// CHECK-LABEL: func @accept_different_alloc_ops
// CHECK-COUNT-3: scf.for
// CHECK-COUNT-2:   linalg.matmul
