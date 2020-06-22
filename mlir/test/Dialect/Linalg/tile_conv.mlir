// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=2,3,0,0,4" | FileCheck %s -check-prefix=TILE-23004

// TILE-23004-DAG: #[[$D0x30pS0x10:.*]] = affine_map<(d0) -> (d0 * 30)>
// TILE-23004-DAG: #[[$S0x10p90D0x30pS1:.*]] = affine_map<(d0)[s0, s1] -> (s0 * 10 + 90, d0 * -30 + s1)>
// TILE-23004-DAG: #[[$strided4D:.*]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
// TILE-23004-DAG: #[[$bound_map_4:.*]] = affine_map<(d0)[s0] -> (4, -d0 + s0)>

func @conv(%arg0: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg1: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg2: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [10, 20], strides = [30, 40]} : memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>
  return
}
// TILE-23004-LABEL: func @conv(
//       TILE-23004:   %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>) {
//       TILE-23004-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-23004-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-23004-DAG: %[[C3:.*]] = constant 3 : index
//       TILE-23004-DAG: %[[C4:.*]] = constant 4 : index
//       TILE-23004:   %[[Q:.*]] = dim %{{.*}}, %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:   %[[B:.*]] = dim %{{.*}}, %c0 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:   %[[PaddedInput0:.*]] = dim %{{.*}}, %c1 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:   %[[X0:.*]] = dim %{{.*}}, %c1 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:   scf.for %[[ivI:.*]] = %{{.*}} to %[[B]] step %{{.*}} {
//       TILE-23004:     scf.for %[[ivJ:.*]] = %{{.*}} to %[[X0]] step %{{.*}} {
//       TILE-23004:       scf.for %[[ivK:.*]] = %{{.*}} to %[[Q]] step %{{.*}} {
//       TILE-23004:         %[[Z0:.*]] = dim %{{.*}}, %c0 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:         %[[Z1:.*]] = dim %{{.*}}, %c1 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:         %[[Z2:.*]] = dim %{{.*}}, %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:         %[[szK:.*]] = affine.min #[[$bound_map_4]](%[[ivK]])[%[[Z2]]]
//       TILE-23004:         %[[K:.*]] = dim %{{.*}}, %c3 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:         %[[FilterView:.*]] = subview %{{.*}}[0, 0, %[[ivK]], 0] [%[[Z0]], %[[Z1]], %[[szK]], %[[K]]] [1, 1, 1, 1] : memref<?x?x?x?xf32, #[[$strided4D]]> to memref<?x?x?x?xf32, #[[$strided4D]]>
//
//       TILE-23004:         %[[J1:.*]] = affine.apply #[[$D0x30pS0x10]](%[[ivJ]])
//       TILE-23004:         %[[PaddedInput0b:.*]] = dim %{{.*}}, %c1 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:         %[[I1pStep:.*]] = affine.min #[[$S0x10p90D0x30pS1]](%[[ivJ]])[%[[PaddedInput0]], %[[PaddedInput0b]]]
//       TILE-23004:         %[[SZ2:.*]] = dim %{{.*}}, %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:         %[[dim3:.*]] = dim %{{.*}}, %c3
//       TILE-23004:         %[[sz3:.*]] = affine.min #[[$bound_map_4]](%[[ivK]])[%[[dim3]]]
//       TILE-23004:         %[[InputView:.*]] = subview %{{.*}}[%[[ivI]], %[[J1]], 0, %[[ivK]]] [%{{.*}}, %{{.*}}, %[[SZ2]], %[[sz3]]] [1, 1, 1, 1] : memref<?x?x?x?xf32, #[[$strided4D]]> to memref<?x?x?x?xf32, #[[$strided4D]]>
//
//       TILE-23004:         %[[X0:.*]] = dim %{{.*}}, %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:         %[[X1:.*]] = dim %{{.*}}, %c3 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       TILE-23004:         %[[OutputView:.*]] = subview %{{.*}}[%[[ivI]], %[[ivJ]], 0, 0] [%{{.*}}, %{{.*}}, %[[X0]], %[[X1]]] [1, 1, 1, 1] : memref<?x?x?x?xf32, #[[$strided4D]]> to memref<?x?x?x?xf32, #[[$strided4D]]>
//
//       TILE-23004:         linalg.conv(%[[FilterView]], %[[InputView]], %[[OutputView]]) {dilations = [10, 20], strides = [30, 40]} : memref<?x?x?x?xf32, #[[$strided4D]]>, memref<?x?x?x?xf32, #[[$strided4D]]>, memref<?x?x?x?xf32, #[[$strided4D]]>
