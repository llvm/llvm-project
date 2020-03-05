// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2 | FileCheck %s -check-prefix=TILE-2
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=0,2 | FileCheck %s -check-prefix=TILE-02
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=0,0,2 | FileCheck %s -check-prefix=TILE-002
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2,3,4 | FileCheck %s -check-prefix=TILE-234

//   TILE-2-DAG: #[[strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
//  TILE-02-DAG: #[[strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// TILE-002-DAG: #[[strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// TILE-234-DAG: #[[strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

//   TILE-2-DAG: #[[strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
//  TILE-02-DAG: #[[strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// TILE-002-DAG: #[[strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// TILE-234-DAG: #[[strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>

//   TILE-2-DAG: #[[bound_map:.*]] = affine_map<(d0, d1, d2) -> (d0, d1 - d2)>
//  TILE-02-DAG: #[[bound_map:.*]] = affine_map<(d0, d1, d2) -> (d0, d1 - d2)>
// TILE-002-DAG: #[[bound_map:.*]] = affine_map<(d0, d1, d2) -> (d0, d1 - d2)>
// TILE-234-DAG: #[[bound_map:.*]] = affine_map<(d0, d1, d2) -> (d0, d1 - d2)>

//   TILE-2-DAG: #[[strided1D_dynamic:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
//   TILE-02-DAG: #[[strided1D_dynamic:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
//   T_ILE-002-DAG: #[[strided1D_dynamic:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
//   TILE-234-DAG: #[[strided1D_dynamic:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

//   TILE-2-DAG: #[[strided2D_dynamic:.*]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
//   TILE-02-DAG: #[[strided2D_dynamic:.*]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
//   TILE-002-DAG: #[[strided2D_dynamic:.*]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
//   TILE-234-DAG: #[[strided2D_dynamic:.*]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>

//   REACTIVATE_ME_TILE-2-DAG: #[[stride_99_1_layout_map:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 99 + s0 + d1)>
//  REACTIVATE_ME_TILE-02-DAG: #[[stride_99_1_layout_map:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 99 + s0 + d1)>
// REACTIVATE_ME_TILE-234-DAG: #[[stride_99_1_layout_map:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 99 + s0 + d1)>

func @matmul(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg2: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.matmul(%arg0, %arg1, %arg2) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
// TILE-2-LABEL: func @matmul(
//       TILE-2-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-2-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-2-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-2: %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2: loop.for %[[I:.*]] = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[localM:.*]] = dim %{{.*}}, 0
//       TILE-2:   %[[szM:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localM]], %[[I]])
//       TILE-2:   %[[K:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2:   %[[sAi:.*]] = subview %{{.*}}[%[[I]], %[[C0]]] [%[[szM]], %[[K]]] [%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-2:   %[[localK:.*]] = dim %{{.*}}, 0
//       TILE-2:   %[[szK:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localK]], %[[I]])
//       TILE-2:   %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2:   %[[sCi:.*]] = subview %{{.*}}[%[[I]], %[[C0]]] [%[[szK]], %[[N]]] [%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-2:   linalg.matmul(%[[sAi]], %{{.*}}, %[[sCi]]) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D_dynamic]]>

// TILE-02-LABEL: func @matmul(
//       TILE-02-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-02-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-02-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-02: %[[N:.*]] = dim %arg1, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02: loop.for %[[J:.*]] = %{{.*}} to %[[N]] step %{{.*}} {
//       TILE-02:   %[[K:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02:   %[[localN:.*]] = dim %{{.*}}, 1
//       TILE-02:   %[[szN:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localN]], %[[J]])
//       TILE-02:   %[[sBj:.*]] = subview %{{.*}}[%[[C0]], %[[J]]] [%[[K]], %[[szN]]] [%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-02:   %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02:   %[[localK:.*]] = dim %{{.*}}, 1
//       TILE-02:   %[[szK:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localK]], %[[J]])
//       TILE-02:   %[[sCj:.*]] = subview %{{.*}}[%[[C0]], %[[J]]] [%[[M]], %[[szK]]] [%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-02:   linalg.matmul(%{{.*}}, %[[sBj]], %[[sCj]]) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D_dynamic]]>

// TILE-002-LABEL: func @matmul(
//       TILE-002-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-002-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-002-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-002: %[[ubK:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-002: loop.for %[[K:.*]] = %{{.*}}{{.*}} to %[[ubK]] step %{{.*}} {
//       TILE-002:   %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-002:   %[[localK:.*]] = dim %{{.*}}, 1
//       TILE-002:   %[[szK:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localK]], %[[K]])
//       TILE-002:   %[[sAj:.*]] = subview %{{.*}}[%[[C0]], %[[K]]] [%[[M]], %[[szK]]] [%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-002:   %[[localK:.*]] = dim %{{.*}}, 0
//       TILE-002:   %[[szK:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localK]], %[[K]])
//       TILE-002:   %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-002:   %[[sBj:.*]] = subview %{{.*}}[%[[K]], %[[C0]]] [%[[szK]], %[[N]]] [%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-002:   linalg.matmul(%[[sAj]], %[[sBj]], %{{.*}}) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D]]>

// TILE-234-LABEL: func @matmul(
//       TILE-234-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-234-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-234-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-234-DAG: %[[C3:.*]] = constant 3 : index
//       TILE-234-DAG: %[[C4:.*]] = constant 4 : index
//       TILE-234: %[[ubM:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234: %[[ubK:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234: %[[ubN:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234:  loop.for %[[I:.*]] = %{{.*}}{{.*}} to %[[ubM]] step %{{.*}} {
//       TILE-234:    loop.for %[[J:.*]] = %{{.*}}{{.*}} to %[[ubN]] step %{{.*}} {
//       TILE-234:      loop.for %[[K:.*]] = %{{.*}}{{.*}} to %[[ubK]] step %{{.*}} {
//       TILE-234:        %[[localM:.*]] = dim %{{.*}}, 0
//       TILE-234:        %[[szM:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localM]], %[[I]])
//       TILE-234:        %[[localK:.*]] = dim %{{.*}}, 1
//       TILE-234:        %[[szK:.*]] = affine.min #[[bound_map]](%[[C4]], %[[localK]], %[[K]])
//       TILE-234:        %[[sAik:.*]] = subview %{{.*}}[%[[I]], %[[K]]] [%[[szM]], %[[szK]]] [%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-234:        %[[localK:.*]] = dim %{{.*}}, 0
//       TILE-234:        %[[szK:.*]] = affine.min #[[bound_map]](%[[C4]], %[[localK]], %[[K]])
//       TILE-234:        %[[localN:.*]] = dim %{{.*}}, 1
//       TILE-234:        %[[szN:.*]] = affine.min #[[bound_map]](%[[C3]], %[[localN]], %[[J]])
//       TILE-234:        %[[sBkj:.*]] = subview %{{.*}}[%[[K]], %[[J]]] [%[[szK]], %[[szN]]] [%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-234:        %[[localM:.*]] = dim %{{.*}}, 0
//       TILE-234:        %[[szM:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localM]], %[[I]])
//       TILE-234:        %[[localN:.*]] = dim %{{.*}}, 1
//       TILE-234:        %[[szN:.*]] = affine.min #[[bound_map]](%[[C3]], %[[localN]], %[[J]])
//       TILE-234:        %[[sCij:.*]] = subview %{{.*}}[%[[I]], %[[J]]] [%[[szM]], %[[szN]]] [%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//
//       TILE-234:        linalg.matmul(%[[sAik]], %[[sBkj]], %[[sCij]]) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D_dynamic]]>

// When the buffer shapes are known at compile time, it is possible to avoid
// the "min" in subview size computation. This test uses buffer sizes divisible
// by respective tile sizes (M=10 divisble by 2, N=12 divisible by 2 and 3,
// K=16 divisble by 2 and 4).
func @matmul_static(%arg0: memref<10x16xf32, offset: ?, strides: [?, 1]>, %arg1: memref<16x12xf32, offset: ?, strides: [?, 1]>, %arg2: memref<10x12xf32, offset: ?, strides: [?, 1]>) {
  linalg.matmul(%arg0, %arg1, %arg2) : memref<10x16xf32, offset: ?, strides: [?, 1]>, memref<16x12xf32, offset: ?, strides: [?, 1]>, memref<10x12xf32, offset: ?, strides: [?, 1]>
  return
}
// TILE-2-LABEL: func @matmul_static(
//       TILE-2-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-2-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-2-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-2: %[[M:.*]] = dim %{{.*}}, 0 : memref<10x16xf32, #[[strided2D]]>
//       TILE-2: loop.for %[[I:.*]] = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[K:.*]] = dim %{{.*}}, 1 : memref<10x16xf32, #[[strided2D]]>
//       TILE-2:   %[[sAi:.*]] = subview %{{.*}}[%[[I]], %[[C0]]] [%[[C2]], %[[K]]] [%[[C1]], %[[C1]]] : memref<10x16xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-2:   %[[N:.*]] = dim %{{.*}}, 1 : memref<10x12xf32, #[[strided2D]]>
//       TILE-2:   %[[sCi:.*]] = subview %{{.*}}[%[[I]], %[[C0]]] [%[[C2]], %[[N]]] [%[[C1]], %[[C1]]] : memref<10x12xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-2:   linalg.matmul(%[[sAi]], %{{.*}}, %[[sCi]])

// TILE-02-LABEL: func @matmul_static(
//       TILE-02-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-02-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-02-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-02: %[[N:.*]] = dim %arg1, 1 : memref<16x12xf32, #[[strided2D]]>
//       TILE-02: loop.for %[[J:.*]] = %{{.*}} to %[[N]] step %{{.*}} {
//       TILE-02:   %[[K:.*]] = dim %{{.*}}, 0 : memref<16x12xf32, #[[strided2D]]>
//   TILE-02-NOT:   affine.min
//       TILE-02:   %[[sBj:.*]] = subview %{{.*}}[%[[C0]], %[[J]]] [%[[K]], %[[C2]]] [%[[C1]], %[[C1]]] : memref<16x12xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-02:   %[[M:.*]] = dim %{{.*}}, 0 : memref<10x12xf32, #[[strided2D]]>
//   TILE-02-NOT:   affine.min
//       TILE-02:   %[[sCj:.*]] = subview %{{.*}}[%[[C0]], %[[J]]] [%[[M]], %[[C2]]] [%[[C1]], %[[C1]]] : memref<10x12xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-02:   linalg.matmul(%{{.*}}, %[[sBj]], %[[sCj]]) : memref<10x16xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D_dynamic]]>

// TILE-002-LABEL: func @matmul_static(
//       TILE-002-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-002-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-002-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-002: %[[ubK:.*]] = dim %{{.*}}, 1 : memref<10x16xf32, #[[strided2D]]>
//       TILE-002: loop.for %[[K:.*]] = %{{.*}}{{.*}} to %[[ubK]] step %{{.*}} {
//       TILE-002:   %[[M:.*]] = dim %{{.*}}, 0 : memref<10x16xf32, #[[strided2D]]>
//   TILE-002-NOT:   affine.min
//       TILE-002:   %[[sAj:.*]] = subview %{{.*}}[%[[C0]], %[[K]]] [%[[M]], %[[C2]]] [%[[C1]], %[[C1]]] : memref<10x16xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-002:   %[[N:.*]] = dim %{{.*}}, 1 : memref<16x12xf32, #[[strided2D]]>
//   TILE-002-NOT:   affine.min
//       TILE-002:   %[[sBj:.*]] = subview %{{.*}}[%[[K]], %[[C0]]] [%[[C2]], %[[N]]] [%[[C1]], %[[C1]]] : memref<16x12xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-002:   linalg.matmul(%[[sAj]], %[[sBj]], %{{.*}}) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D_dynamic]]>, memref<10x12xf32, #[[strided2D]]>

// TILE-234-LABEL: func @matmul_static(
//       TILE-234-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-234-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-234-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-234-DAG: %[[C3:.*]] = constant 3 : index
//       TILE-234-DAG: %[[C4:.*]] = constant 4 : index
//       TILE-234: %[[ubM:.*]] = dim %{{.*}}, 0 : memref<10x16xf32, #[[strided2D]]>
//       TILE-234: %[[ubK:.*]] = dim %{{.*}}, 1 : memref<10x16xf32, #[[strided2D]]>
//       TILE-234: %[[ubN:.*]] = dim %{{.*}}, 1 : memref<16x12xf32, #[[strided2D]]>
//       TILE-234:  loop.for %[[I:.*]] = %{{.*}}{{.*}} to %[[ubM]] step %{{.*}} {
//       TILE-234:    loop.for %[[J:.*]] = %{{.*}}{{.*}} to %[[ubN]] step %{{.*}} {
//       TILE-234:      loop.for %[[K:.*]] = %{{.*}}{{.*}} to %[[ubK]] step %{{.*}} {
//   TILE-234-NOT:   affine.min
//       TILE-234:        %[[sAik:.*]] = subview %{{.*}}[%[[I]], %[[K]]] [%[[C2]], %[[C4]]] [%[[C1]], %[[C1]]] : memref<10x16xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//   TILE-234-NOT:   affine.min
//       TILE-234:        %[[sBkj:.*]] = subview %{{.*}}[%[[K]], %[[J]]] [%[[C4]], %[[C3]]] [%[[C1]], %[[C1]]] : memref<16x12xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//   TILE-234-NOT:   affine.min
//       TILE-234:        %[[sCij:.*]] = subview %{{.*}}[%[[I]], %[[J]]] [%[[C2]], %[[C3]]] [%[[C1]], %[[C1]]] : memref<10x12xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//
//       TILE-234:        linalg.matmul(%[[sAik]], %[[sBkj]], %[[sCij]]) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D_dynamic]]>

func @matvec(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.matvec(%arg0, %arg1, %arg2) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>
  return
}
// TILE-2-LABEL: func @matvec(
//       TILE-2-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-2-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-2-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-2: %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2: loop.for %[[I:.*]] = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[localM:.*]] = dim %{{.*}}, 0
//       TILE-2:   %[[szM:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localM]], %[[I]])
//       TILE-2:   %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2:   %[[sAi:.*]] = subview %{{.*}}[%[[I]], %[[C0]]] [%[[szM]], %[[N]]] [%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-2:   %[[localN:.*]] = dim %{{.*}}, 0
//       TILE-2:   %[[szN:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localN]], %[[I]])
//       TILE-2:   %[[sCi:.*]] = subview %{{.*}}[%[[I]]] [%[[szN]]] [%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//       TILE-2:   linalg.matvec(%[[sAi]], %{{.*}}, %[[sCi]]) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?xf32, #[[strided1D]]>, memref<?xf32, #[[strided1D_dynamic]]>

// TILE-02-LABEL: func @matvec(
//       TILE-02-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-02-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-02-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-02: %[[K:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02: loop.for %[[J]] = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-02:   %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02:   %[[localN:.*]] = dim %{{.*}}, 1
//       TILE-02:   %[[szN:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localN]], %[[J]])
//       TILE-02:   %[[sAj:.*]] = subview %{{.*}}[%[[C0]], %[[J]]] [%[[M]], %[[szN]]] [%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-02:   %[[localN:.*]] = dim %{{.*}}, 0
//       TILE-02:   %[[szN:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localN]], %[[J]])
//       TILE-02:   %[[sBj:.*]] = subview %{{.*}}[%[[J]]] [%[[szN]]] [%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//       TILE-02:   linalg.matvec(%[[sAj]], %[[sBj]], %{{.*}}) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?xf32, #[[strided1D_dynamic]]>, memref<?xf32, #[[strided1D]]>

// TILE-002-LABEL: func @matvec(
//   TILE-002-NOT: loop.for

// TILE-234-LABEL: func @matvec(
//       TILE-234-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-234-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-234-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-234-DAG: %[[C3:.*]] = constant 3 : index
//       TILE-234: %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234: %[[K:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234:  loop.for %[[I:.*]] = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-234:    loop.for %[[J:.*]] = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-234:      %[[localM:.*]] = dim %{{.*}}, 0
//       TILE-234:      %[[szM:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localM]], %[[I]])
//       TILE-234:      %[[localN:.*]] = dim %{{.*}}, 1
//       TILE-234:      %[[szN:.*]] = affine.min #[[bound_map]](%[[C3]], %[[localN]], %[[J]])
//       TILE-234:      %[[sAij:.*]] = subview %{{.*}}[%[[I]], %[[J]]] [%[[szM]], %[[szN]]] [%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-234:      %[[localN:.*]] = dim %{{.*}}, 0
//       TILE-234:      %[[szN:.*]] = affine.min #[[bound_map]](%[[C3]], %[[localN]], %[[J]])
//       TILE-234:      %[[sBj:.*]] = subview %{{.*}}[%[[J]]] [%[[szN]]] [%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//       TILE-234:      %[[localM:.*]] = dim %{{.*}}, 0
//       TILE-234:      %[[szM:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localM]], %[[I]])
//       TILE-234:      %[[sCi:.*]] = subview %{{.*}}[%[[I]]] [%[[szM]]] [%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//
//       TILE-234:      linalg.matvec(%[[sAij]], %[[sBj]], %[[sCi]]) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?xf32, #[[strided1D_dynamic]]>, memref<?xf32, #[[strided1D_dynamic]]>

func @dot(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>, memref<f32>
  return
}
// TILE-2-LABEL: func @dot(
//       TILE-2-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-2-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-2-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-2: %[[M:.*]] = dim %{{.*}}, 0 : memref<?xf32, #[[strided1D]]>
//       TILE-2: loop.for %[[I:.*]] = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[localM:.*]] = dim %{{.*}}, 0
//       TILE-2:   %[[szM:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localM]], %[[I]])
//       TILE-2:   %[[sAi:.*]] = subview %{{.*}}[%[[I]]] [%[[szM]]] [%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//       TILE-2:   %[[localM:.*]] = dim %{{.*}}, 0
//       TILE-2:   %[[szM:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localM]], %[[I]])
//       TILE-2:   %[[sBi:.*]] = subview %{{.*}}[%[[I]]] [%[[szM]]] [%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//       TILE-2:   linalg.dot(%[[sAi]], %[[sBi]], {{.*}}) : memref<?xf32, #[[strided1D_dynamic]]>, memref<?xf32, #[[strided1D_dynamic]]>, memref<f32>

// TILE-02-LABEL: func @dot(
//   TILE-02-NOT: loop.for

// TILE-002-LABEL: func @dot(
//   TILE-002-NOT: loop.for

// TILE-234-LABEL: func @dot(
//       TILE-234-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-234-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-234-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-234:  %[[ubK:.*]] = dim %{{.*}}, 0 : memref<?xf32, #[[strided1D]]>
//       TILE-234:  loop.for %[[I:.*]] = %{{.*}} to %[[ubK]] step %{{.*}} {
//       TILE-234:    %[[localM:.*]] = dim %{{.*}}, 0
//       TILE-234:    %[[szM:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localM]], %[[I]])
//       TILE-234:    %[[sAi:.*]] = subview %{{.*}}[%[[I]]] [%[[szM]]] [%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//       TILE-234:    %[[localM:.*]] = dim %{{.*}}, 0
//       TILE-234:    %[[szM:.*]] = affine.min #[[bound_map]](%[[C2]], %[[localM]], %[[I]])
//       TILE-234:    %[[sBi:.*]] = subview %{{.*}}[%[[I]]] [%[[szM]]] [%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//       TILE-234:    linalg.dot(%[[sAi]], %[[sBi]], %{{.*}}) : memref<?xf32, #[[strided1D_dynamic]]>, memref<?xf32, #[[strided1D_dynamic]]>, memref<f32>

func @fill_static(%arg0: memref<127x99xf32>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<127x99xf32>, f32
  return
}
// TILE-2-LABEL: func @fill_static
//       TILE-2:   for
//   TILE-2-NOT:   for
//       TILE-2:       subview{{.*}} : memref<127x99xf32>
//       TILE-2:       linalg.fill{{.*}} : memref<?x?xf32, #[[strided2D_dynamic]]>, f32

// TILE-02-LABEL: func @fill_static
//       TILE-02:   for
//   TILE-02-NOT:   for
//       TILE-02:       subview{{.*}} : memref<127x99xf32>
//       TILE-02:       linalg.fill{{.*}} : memref<?x?xf32, #[[strided2D_dynamic]]>, f32

// TILE-002-LABEL: func @fill_static
//   TILE-002-NOT:   for
//       TILE-002:     linalg.fill{{.*}} memref<127x99xf32>, f32

// TILE-234-LABEL: func @fill_static
//       TILE-234:   for
//       TILE-234:     for
//   TILE-234-NOT:   for
//       TILE-234:       subview{{.*}} : memref<127x99xf32>
//       TILE-234:       linalg.fill{{.*}} : memref<?x?xf32, #[[strided2D_dynamic]]>, f32


func @fill(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<?x?xf32, offset: ?, strides: [?, 1]>, f32
  return
}
// TILE-2-LABEL: func @fill
//       TILE-2:   for
//   TILE-2-NOT:   for
//       TILE-2:   fill{{.*}} f32

// TILE-02-LABEL: func @fill
//       TILE-02:   for
//   TILE-02-NOT:   for
//       TILE-02:     fill{{.*}} f32

// TILE-002-LABEL: func @fill
//   TILE-002-NOT:   for
//       TILE-002:     fill{{.*}} f32

// TILE-234-LABEL: func @fill
//       TILE-234:   for
//       TILE-234:     for
//   TILE-234-NOT:   for
//       TILE-234:       fill{{.*}} f32

#id_2d = affine_map<(i, j) -> (i, j)>
#pointwise_2d_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  iterator_types = ["parallel", "parallel"]
}

func @pointwise(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                %arg2: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.generic #pointwise_2d_trait %arg0, %arg1, %arg2 {
  ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):   // no predecessors
    %4 = addf %arg4, %arg5 : f32
    linalg.yield %4 : f32
  }: memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
// TILE-2-LABEL: func @pointwise
//       TILE-2:   for
//   TILE-2-NOT:   for
//       TILE-2:   linalg.generic

// TILE-02-LABEL: func @pointwise
//       TILE-02:   for
//   TILE-02-NOT:   for
//       TILE-02:     linalg.generic

// TILE-002-LABEL: func @pointwise
//   TILE-002-NOT:   for
//       TILE-002:     linalg.generic

// TILE-234-LABEL: func @pointwise
//       TILE-234:   for
//       TILE-234:     for
//   TILE-234-NOT:   for
//       TILE-234:       linalg.generic
