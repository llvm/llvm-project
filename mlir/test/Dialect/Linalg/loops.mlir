// RUN: mlir-opt %s -convert-linalg-to-loops | FileCheck --check-prefix=CHECKLOOP %s
// RUN: mlir-opt %s -convert-linalg-to-parallel-loops | FileCheck --check-prefix=CHECKPARALLEL %s

// Test that we can lower all the way to LLVM without crashing, don't check results here.
// RUN: mlir-opt %s -convert-linalg-to-loops -convert-linalg-to-llvm -o=/dev/null 2>&1

// CHECKLOOP-DAG: #[[$strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECKLOOP-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECKLOOP-DAG: #[[$strided3D:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>
// CHECKLOOP-DAG: #[[$strided4D:.*]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
// CHECKLOOP-DAG: #[[$clampMinMap:.*]] = affine_map<(d0) -> (d0, 0)>

// CHECKLOOP-DAG: #[[$stride1Dilation1:.*]] = affine_map<(d0, d1) -> (d0  + d1)>
// CHECKLOOP-DAG: #[[$stride2Dilation1:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECKLOOP-DAG: #[[$stride2Dilation4:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1 * 4)>
// CHECKLOOP-DAG: #[[$stride3Dilation5:.*]] = affine_map<(d0, d1) -> (d0 * 3 + d1 * 5)>
// CHECKLOOP-DAG: #[[$convMap:.*]] = affine_map<(d0, d1)[s0] -> (d0 + d1 - s0 floordiv 2)>

// CHECKPARALLEL-DAG: #[[$strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECKPARALLEL-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECKPARALLEL-DAG: #[[$strided3D:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>
// CHECKPARALLEL-DAG: #[[$strided4D:.*]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
// CHECKPARALLEL-DAG: #[[$clampMinMap:.*]] = affine_map<(d0) -> (d0, 0)>

// CHECKPARALLEL-DAG: #[[$stride1Dilation1:.*]] = affine_map<(d0, d1) -> (d0  + d1)>
// CHECKPARALLEL-DAG: #[[$stride2Dilation1:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECKPARALLEL-DAG: #[[$stride2Dilation4:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1 * 4)>
// CHECKPARALLEL-DAG: #[[$stride3Dilation5:.*]] = affine_map<(d0, d1) -> (d0 * 3 + d1 * 5)>
// CHECKPARALLEL-DAG: #[[$convMap:.*]] = affine_map<(d0, d1)[s0] -> (d0 + d1 - s0 floordiv 2)>


func @matmul(%arg0: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %A = view %arg0[%c0][%M, %K] : memref<?xi8> to memref<?x?xf32>
  %B = view %arg0[%c0][%K, %N] : memref<?xi8> to memref<?x?xf32>
  %C = view %arg0[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32>
  linalg.matmul %A, %B, %C : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
  return
}
// CHECKLOOP-LABEL: func @matmul(%{{.*}}: memref<?xi8>,
//  CHECKLOOP-SAME: [[M:arg[0-9]+]]: index
//  CHECKLOOP-SAME: [[N:arg[0-9]+]]: index
//  CHECKLOOP-SAME: [[K:arg[0-9]+]]: index
//       CHECKLOOP: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECKLOOP: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECKLOOP: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECKLOOP: scf.for %{{.*}} = %{{.*}} to %[[M]] step %{{.*}} {
//       CHECKLOOP:   scf.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} {
//       CHECKLOOP:     scf.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKLOOP-DAG:       %[[a:.*]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECKLOOP-DAG:       %[[b:.*]] = load %[[B]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECKLOOP-DAG:       %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKLOOP-DAG:       %[[c:.*]] = load %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECKLOOP-DAG:       %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKLOOP:       store %[[res]], %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32>

// CHECKPARALLEL-LABEL: func @matmul(%{{.*}}: memref<?xi8>,
//  CHECKPARALLEL-SAME: [[M:arg[0-9]+]]: index
//  CHECKPARALLEL-SAME: [[N:arg[0-9]+]]: index
//  CHECKPARALLEL-SAME: [[K:arg[0-9]+]]: index
//       CHECKPARALLEL: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECKPARALLEL: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECKPARALLEL: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECKPARALLEL: scf.parallel (%{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}) to (%[[M]], %[[N]]) step (%{{.*}}, %{{.*}} {
//       CHECKPARALLEL:   scf.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKPARALLEL-DAG:     %[[a:.*]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECKPARALLEL-DAG:     %[[b:.*]] = load %[[B]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECKPARALLEL-DAG:     %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKPARALLEL-DAG:     %[[c:.*]] = load %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECKPARALLEL-DAG:     %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKPARALLEL:     store %[[res]], %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32>



func @matvec(%arg0: memref<?xi8>, %M: index, %N: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %2 = view %arg0[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32>
  %3 = view %arg0[%c0][%M] : memref<?xi8> to memref<?xf32>
  %4 = view %arg0[%c0][%N] : memref<?xi8> to memref<?xf32>
  linalg.matvec %2, %3, %4 : (memref<?x?xf32>, memref<?xf32>, memref<?xf32>)
  return
}
// CHECKLOOP-LABEL: func @matvec(%{{.*}}: memref<?xi8>,
//  CHECKLOOP-SAME: [[M:arg[0-9]+]]: index
//  CHECKLOOP-SAME: [[K:arg[0-9]+]]: index
//       CHECKLOOP: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECKLOOP: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECKLOOP: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECKLOOP: scf.for %{{.*}} = %{{.*}} to %[[M]] step %{{.*}} {
//       CHECKLOOP:   scf.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKLOOP-DAG:     %[[a:.*]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECKLOOP-DAG:     %[[b:.*]] = load %[[B]][%{{.*}}] : memref<?xf32>
//   CHECKLOOP-DAG:     %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKLOOP-DAG:     %[[c:.*]] = load %[[C]][%{{.*}}] : memref<?xf32>
//   CHECKLOOP-DAG:     %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKLOOP:     store %[[res]], %[[C]][%{{.*}}] : memref<?xf32>

// CHECKPARALLEL-LABEL: func @matvec(%{{.*}}: memref<?xi8>,
//  CHECKPARALLEL-SAME: [[M:arg[0-9]+]]: index
//  CHECKPARALLEL-SAME: [[K:arg[0-9]+]]: index
//       CHECKPARALLEL: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECKPARALLEL: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECKPARALLEL: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECKPARALLEL: scf.parallel (%{{.*}}) = (%{{.*}}) to (%[[M]]) step (%{{.*}}) {
//       CHECKPARALLEL:   scf.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKPARALLEL-DAG:     %[[a:.*]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECKPARALLEL-DAG:     %[[b:.*]] = load %[[B]][%{{.*}}] : memref<?xf32>
//   CHECKPARALLEL-DAG:     %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKPARALLEL-DAG:     %[[c:.*]] = load %[[C]][%{{.*}}] : memref<?xf32>
//   CHECKPARALLEL-DAG:     %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKPARALLEL:     store %[[res]], %[[C]][%{{.*}}] : memref<?xf32>


func @dot(%arg0: memref<?xi8>, %M: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %1 = view %arg0[%c0][%M] : memref<?xi8> to memref<?xf32>
  %2 = view %arg0[%c0][%M] : memref<?xi8> to memref<?xf32>
  %3 = view %arg0[%c0][] : memref<?xi8> to memref<f32>
  linalg.dot(%1, %2, %3) : memref<?xf32>, memref<?xf32>, memref<f32>
  return
}
// CHECKLOOP-LABEL: func @dot(%{{.*}}: memref<?xi8>,
//  CHECKLOOP-SAME: [[K:arg[0-9]+]]: index
//       CHECKLOOP: %[[A:.*]] = std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECKLOOP: %[[B:.*]] = std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECKLOOP: %[[C:.*]] = std.view %{{.*}}[{{.*}}][] : memref<?xi8> to memref<f32>
//       CHECKLOOP: scf.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKLOOP-DAG:   %[[a:.*]] = load %[[A]][%{{.*}}] : memref<?xf32>
//   CHECKLOOP-DAG:   %[[b:.*]] = load %[[B]][%{{.*}}] : memref<?xf32>
//   CHECKLOOP-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKLOOP-DAG:   %[[c:.*]] = load %[[C]][] : memref<f32>
//   CHECKLOOP-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKLOOP:   store %[[res]], %[[C]][] : memref<f32>

// CHECKPARALLEL-LABEL: func @dot(%{{.*}}: memref<?xi8>,
//  CHECKPARALLEL-SAME: [[K:arg[0-9]+]]: index
//       CHECKPARALLEL: %[[A:.*]] = std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECKPARALLEL: %[[B:.*]] = std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECKPARALLEL: %[[C:.*]] = std.view %{{.*}}[{{.*}}][] : memref<?xi8> to memref<f32>
//       CHECKPARALLEL: scf.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKPARALLEL-DAG:   %[[a:.*]] = load %[[A]][%{{.*}}] : memref<?xf32>
//   CHECKPARALLEL-DAG:   %[[b:.*]] = load %[[B]][%{{.*}}] : memref<?xf32>
//   CHECKPARALLEL-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKPARALLEL-DAG:   %[[c:.*]] = load %[[C]][] : memref<f32>
//   CHECKPARALLEL-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKPARALLEL:   store %[[res]], %[[C]][] : memref<f32>


func @dot_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>, memref<f32>
  return
}
// CHECKLOOP-LABEL: func @dot_view(
//       CHECKLOOP:   %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: memref<f32>) {
//       CHECKLOOP: %[[K:.*]] = dim %arg0, %c0 : memref<?xf32, #[[$strided1D]]>
//       CHECKLOOP: scf.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKLOOP-DAG:   %[[a:.*]] = load %arg0[%{{.*}}] : memref<?xf32, #[[$strided1D]]>
//   CHECKLOOP-DAG:   %[[b:.*]] = load %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>
//   CHECKLOOP-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKLOOP-DAG:   %[[c:.*]] = load %{{.*}}[] : memref<f32>
//   CHECKLOOP-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKLOOP:   store %[[res]], %{{.*}}[] : memref<f32>

// CHECKPARALLEL-LABEL: func @dot_view(
//       CHECKPARALLEL:   %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: memref<f32>) {
//       CHECKPARALLEL: %[[K:.*]] = dim %arg0, %c0 : memref<?xf32, #[[$strided1D]]>
//       CHECKPARALLEL: scf.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKPARALLEL-DAG:   %[[a:.*]] = load %arg0[%{{.*}}] : memref<?xf32, #[[$strided1D]]>
//   CHECKPARALLEL-DAG:   %[[b:.*]] = load %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>
//   CHECKPARALLEL-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKPARALLEL-DAG:   %[[c:.*]] = load %{{.*}}[] : memref<f32>
//   CHECKPARALLEL-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKPARALLEL:   store %[[res]], %{{.*}}[] : memref<f32>

func @fill_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<?xf32, offset: ?, strides: [1]>, f32
  return
}
// CHECKLOOP-LABEL: func @fill_view(
//       CHECKLOOP: %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: f32) {
//       CHECKLOOP:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:     store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>

// CHECKPARALLEL-LABEL: func @fill_view(
//       CHECKPARALLEL: %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: f32) {
//       CHECKPARALLEL:   scf.parallel (%{{.*}}) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
//       CHECKPARALLEL:     store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>

func @fill_view0(%arg0: memref<f32>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<f32>, f32
  return
}
// CHECKLOOP-LABEL: func @fill_view0(%{{.*}}: memref<f32>, %{{.*}}: f32) {
//       CHECKLOOP:   store %{{.*}}, %{{.*}}[] : memref<f32>

// CHECKPARALLEL-LABEL: func @fill_view0(%{{.*}}: memref<f32>, %{{.*}}: f32) {
//       CHECKPARALLEL:   store %{{.*}}, %{{.*}}[] : memref<f32>

func @fill_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, f32
  return
}
// CHECKLOOP-LABEL: func @fill_view3(
//       CHECKLOOP: %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: f32) {
//       CHECKLOOP:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:         store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>

// CHECKPARALLEL-LABEL: func @fill_view3(
//       CHECKPARALLEL: %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: f32) {
//       CHECKPARALLEL:   scf.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>

func @copy_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.copy(%arg0, %arg1) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECKLOOP-LABEL: func @copy_view(
//       CHECKLOOP: %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: memref<?xf32, #[[$strided1D]]>) {
//       CHECKLOOP:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:     %[[L:.*]] = load %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>
//       CHECKLOOP:     store %[[L]], %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>

// CHECKPARALLEL-LABEL: func @copy_view(
//       CHECKPARALLEL: %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: memref<?xf32, #[[$strided1D]]>) {
//       CHECKPARALLEL:   scf.parallel (%{{.*}}) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
//       CHECKPARALLEL:     %[[L:.*]] = load %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>
//       CHECKPARALLEL:     store %[[L]], %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>

func @copy_view0(%arg0: memref<f32>, %arg1: memref<f32>) {
  linalg.copy(%arg0, %arg1) : memref<f32>, memref<f32>
  return
}
// CHECKLOOP-LABEL: func @copy_view0(%{{.*}}: memref<f32>, %{{.*}}: memref<f32>) {
//       CHECKLOOP:   %{{.*}} = load %{{.*}}[] : memref<f32>
//       CHECKLOOP:   store %{{.*}}, %{{.*}}[] : memref<f32>

// CHECKPARALLEL-LABEL: func @copy_view0(%{{.*}}: memref<f32>, %{{.*}}: memref<f32>) {
//       CHECKPARALLEL:   %{{.*}} = load %{{.*}}[] : memref<f32>
//       CHECKPARALLEL:   store %{{.*}}, %{{.*}}[] : memref<f32>

func @copy_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.copy(%arg0, %arg1) {inputPermutation = affine_map<(i, j, k) -> (i, k, j)>,
                             outputPermutation = affine_map<(i, j, k) -> (k, j, i)>} :
    memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECKLOOP-LABEL: func @copy_view3
//       CHECKLOOP: (%{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>) {
//       CHECKLOOP:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:         %[[L:.*]] = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKLOOP:         store %[[L]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>

// CHECKPARALLEL-LABEL: func @copy_view3
//       CHECKPARALLEL: (%{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>) {
//       CHECKPARALLEL:   scf.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     %[[L:.*]] = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:     store %[[L]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>

func @conv_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {strides = [2]}: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECKLOOP-LABEL: func @conv_view3(
//       CHECKLOOP: %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>) {
//       CHECKLOOP:   %[[Z0:.*]] = dim %arg0, %c0 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKLOOP:   %[[Q:.*]] = dim %arg0, %c1 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKLOOP:   %[[K:.*]] = dim %arg0, %c2 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKLOOP:   %[[B:.*]] = dim %arg1, %c0 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKLOOP:   %[[X0:.*]] = dim %arg2, %c1 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKLOOP:   scf.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       CHECKLOOP:     scf.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       CHECKLOOP:       scf.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECKLOOP:         scf.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECKLOOP:           scf.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECKLOOP:             %[[SUM:.*]] = affine.apply #[[$stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKLOOP:             %{{.*}} = load %{{.*}}[%{{.*}}, %[[SUM]], %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKLOOP:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKLOOP:             %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKLOOP:             %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:             store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>

// CHECKPARALLEL-LABEL: func @conv_view3(
//       CHECKPARALLEL: %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>) {
//       CHECKPARALLEL:   %[[Z0:.*]] = dim %arg0, %c0 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:   %[[Q:.*]] = dim %arg0, %c1 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:   %[[K:.*]] = dim %arg0, %c2 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:   %[[B:.*]] = dim %arg1, %c0 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:   %[[X0:.*]] = dim %arg2, %c1 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:   scf.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}) to (%[[B]], %[[X0]], %[[K]]) step (%{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     scf.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECKPARALLEL:       scf.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECKPARALLEL:         %[[SUM:.*]] = affine.apply #[[$stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:         %{{.*}} = load %{{.*}}[%{{.*}}, %[[SUM]], %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:         %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:         %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECKPARALLEL:         %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:         %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECKPARALLEL:         store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>

func @conv_view4(%arg0: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg1: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg2: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [4, 5], strides = [2, 3]} : memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>
  return
}
// CHECKLOOP-LABEL: func @conv_view4(
//       CHECKLOOP: %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>) {
//       CHECKLOOP:   %[[Z0:.*]] = dim %arg0, %c0 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKLOOP:   %[[Z1:.*]] = dim %arg0, %c1 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKLOOP:   %[[Q:.*]] = dim %arg0, %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKLOOP:   %[[K:.*]] = dim %arg0, %c3 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKLOOP:   %[[B:.*]] = dim %arg1, %c0 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKLOOP:   %[[X0:.*]] = dim %arg2, %c1 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKLOOP:   %[[X1:.*]] = dim %arg2, %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKLOOP:   scf.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       CHECKLOOP:     scf.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       CHECKLOOP:       scf.for %{{.*}} = %{{.*}} to %[[X1]] step %{{.*}} {
//       CHECKLOOP:         scf.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECKLOOP:           scf.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECKLOOP:             scf.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECKLOOP:               scf.for %{{.*}} = %{{.*}} to %[[Z1]] step %{{.*}} {
//       CHECKLOOP:                 %[[SUM0:.*]] = affine.apply #[[$stride2Dilation4]](%{{.*}}, %{{.*}})
//       CHECKLOOP:                 %[[SUM1:.*]] = affine.apply #[[$stride3Dilation5]](%{{.*}}, %{{.*}})
//       CHECKLOOP:                 %{{.*}} = load %{{.*}}[%{{.*}}, %[[SUM0]], %[[SUM1]], %{{.*}}] : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKLOOP:                 %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKLOOP:                 %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:                 %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKLOOP:                 %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:                 store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[$strided4D]]>

// CHECKPARALLEL-LABEL: func @conv_view4(
//       CHECKPARALLEL: %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>) {
//       CHECKPARALLEL:   %[[Z0:.*]] = dim %arg0, %c0 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:   %[[Z1:.*]] = dim %arg0, %c1 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:   %[[Q:.*]] = dim %arg0, %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:   %[[K:.*]] = dim %arg0, %c3 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:   %[[B:.*]] = dim %arg1, %c0 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:   %[[X0:.*]] = dim %arg2, %c1 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:   %[[X1:.*]] = dim %arg2, %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:   scf.parallel (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) to (%[[B]], %[[X0]], %[[X1]], %[[K]]) step (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     scf.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECKPARALLEL:       scf.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECKPARALLEL:         scf.for %{{.*}} = %{{.*}} to %[[Z1]] step %{{.*}} {
//       CHECKPARALLEL:           %[[SUM0:.*]] = affine.apply #[[$stride2Dilation4]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:           %[[SUM1:.*]] = affine.apply #[[$stride3Dilation5]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:           %{{.*}} = load %{{.*}}[%{{.*}}, %[[SUM0]], %[[SUM1]], %{{.*}}] : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:           %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:           %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECKPARALLEL:           %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:           %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECKPARALLEL:           store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[$strided4D]]>

func @conv_padding(%arg0: memref<?x?x?x?xf32>,
                   %arg1: memref<?x?x?x?xf32>,
                   %arg2: memref<?x?x?x?xf32>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [1, 1],
                                    padding = dense<[[0, 1], [1, 1]]> : tensor<2x2xi64>,
                                    strides = [1, 1]} :
    memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
  return
}
// CHECKLOOP-LABEL: func @conv_padding
//       CHECKLOOP: %{{.*}}: memref<?x?x?x?xf32>, %{{.*}}: memref<?x?x?x?xf32>, %{{.*}}: memref<?x?x?x?xf32>) {
//       CHECKLOOP:   %[[ZERO:.*]] = constant 0.000000e+00 : f32
//       CHECKLOOP:   %[[Z0:.*]] = dim %arg0, %c0 : memref<?x?x?x?xf32>
//       CHECKLOOP:   %[[Z1:.*]] = dim %arg0, %c1 : memref<?x?x?x?xf32>
//       CHECKLOOP:   %[[Q:.*]] =  dim %arg0, %c2 : memref<?x?x?x?xf32>
//       CHECKLOOP:   %[[K:.*]] =  dim %arg0, %c3 : memref<?x?x?x?xf32>
//       CHECKLOOP:   %[[B:.*]] =  dim %arg1, %c0 : memref<?x?x?x?xf32>
//       CHECKLOOP:   %[[X0:.*]] = dim %arg2, %c1 : memref<?x?x?x?xf32>
//       CHECKLOOP:   %[[X1:.*]] = dim %arg2, %c2 : memref<?x?x?x?xf32>
//       CHECKLOOP:   scf.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       CHECKLOOP:     scf.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       CHECKLOOP:       scf.for %{{.*}} = %{{.*}} to %[[X1]] step %{{.*}} {
//       CHECKLOOP:         scf.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECKLOOP:           scf.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECKLOOP:             scf.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECKLOOP:               scf.for %{{.*}} = %{{.*}} to %[[Z1]] step %{{.*}} {
//       CHECKLOOP:                 %[[SUM0:.*]] = affine.apply #{{.*}}(%{{.*}}, %{{.*}})
//       CHECKLOOP:                 %[[SUM1:.*]] = affine.apply #{{.*}}(%{{.*}}, %{{.*}})
//       CHECKLOOP:                 %[[IDX:.*]] = affine.max #[[$clampMinMap]](%[[SUM0]])
//       CHECKLOOP:                 %[[IDY:.*]] = affine.max #[[$clampMinMap]](%[[SUM1]])
//       CHECKLOOP:                 %{{.*}} = load %{{.*}}[%{{.*}}, %[[IDX]], %[[IDY]], %{{.*}}] : memref<?x?x?x?xf32>
//       CHECKLOOP:                 %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:                 %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32>
//       CHECKLOOP:                 %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:                 %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32>
//       CHECKLOOP:                 %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:                 store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32>

// CHECKPARALLEL-LABEL: func @conv_padding
//       CHECKPARALLEL: %{{.*}}: memref<?x?x?x?xf32>, %{{.*}}: memref<?x?x?x?xf32>, %{{.*}}: memref<?x?x?x?xf32>) {
//       CHECKPARALLEL:   %[[ZERO:.*]] = constant 0.000000e+00 : f32
//       CHECKPARALLEL:   %[[Z0:.*]] = dim %arg0, %c0 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[Z1:.*]] = dim %arg0, %c1 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[Q:.*]] =  dim %arg0, %c2 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[K:.*]] =  dim %arg0, %c3 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[B:.*]] =  dim %arg1, %c0 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[X0:.*]] = dim %arg2, %c1 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[X1:.*]] = dim %arg2, %c2 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   scf.parallel (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) to (%[[B]], %[[X0]], %[[X1]], %[[K]]) step (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     scf.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECKPARALLEL:       scf.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECKPARALLEL:         scf.for %{{.*}} = %{{.*}} to %[[Z1]] step %{{.*}} {
//       CHECKPARALLEL:           %[[SUM0:.*]] = affine.apply #{{.*}}(%{{.*}}, %{{.*}})
//       CHECKPARALLEL:           %[[SUM1:.*]] = affine.apply #{{.*}}(%{{.*}}, %{{.*}})
//       CHECKPARALLEL:           %[[IDX:.*]] = affine.max #[[$clampMinMap]](%[[SUM0]])
//       CHECKPARALLEL:           %[[IDY:.*]] = affine.max #[[$clampMinMap]](%[[SUM1]])
//       CHECKPARALLEL:           %{{.*}} = load %{{.*}}[%{{.*}}, %[[IDX]], %[[IDY]], %{{.*}}] : memref<?x?x?x?xf32>
//       CHECKPARALLEL:           %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : f32
//       CHECKPARALLEL:           %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32>
//       CHECKPARALLEL:           %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECKPARALLEL:           %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32>
//       CHECKPARALLEL:           %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECKPARALLEL:           store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32>

func @pooling_max(%arg0: memref<?x?xf32>,
                  %arg1: memref<?x?xi32>,
                  %arg2: memref<?x?xf32>) {
  linalg.pooling_max(%arg0, %arg1, %arg2) { strides = [2, 1] }:
    memref<?x?xf32>, memref<?x?xi32>, memref<?x?xf32>
  return
}
// CHECKLOOP-LABEL: func @pooling_max
//       CHECKLOOP:   %[[WX:.*]] = dim %arg1, %c0 : memref<?x?xi32>
//       CHECKLOOP:   %[[WY:.*]] = dim %arg1, %c1 : memref<?x?xi32>
//       CHECKLOOP:   %[[OX:.*]] = dim %arg2, %c0 : memref<?x?xf32>
//       CHECKLOOP:   %[[OY:.*]] = dim %arg2, %c1 : memref<?x?xf32>
//       CHECKLOOP:   scf.for %{{.*}} = %{{.*}} to %[[OX]] step %{{.*}} {
//       CHECKLOOP:     scf.for %{{.*}} = %{{.*}} to %[[OY]] step %{{.*}} {
//       CHECKLOOP:       scf.for %{{.*}} = %{{.*}} to %[[WX]] step %{{.*}} {
//       CHECKLOOP:         scf.for %{{.*}} = %{{.*}} to %[[WY]] step %{{.*}} {
//       CHECKLOOP:           %[[IX:.*]] = affine.apply #[[$stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKLOOP:           %[[IY:.*]] = affine.apply #[[$stride1Dilation1]](%{{.*}}, %{{.*}})
//       CHECKLOOP:           %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
//       CHECKLOOP:           %{{.*}} = load %{{.*}}[%[[IX]], %[[IY]]] : memref<?x?xf32>
//       CHECKLOOP:           %[[RES:.*]] = select %{{.*}}, %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:           store %[[RES]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>

// CHECKPARALLEL-LABEL: func @pooling_max
//       CHECKPARALLEL:   %[[WX:.*]] = dim %arg1, %c0 : memref<?x?xi32>
//       CHECKPARALLEL:   %[[WY:.*]] = dim %arg1, %c1 : memref<?x?xi32>
//       CHECKPARALLEL:   %[[OX:.*]] = dim %arg2, %c0 : memref<?x?xf32>
//       CHECKPARALLEL:   %[[OY:.*]] = dim %arg2, %c1 : memref<?x?xf32>
//       CHECKPARALLEL:   scf.parallel (%{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}) to (%[[OX]], %[[OY]]) step (%{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     scf.for %{{.*}} = %{{.*}} to %[[WX]] step %{{.*}} {
//       CHECKPARALLEL:       scf.for %{{.*}} = %{{.*}} to %[[WY]] step %{{.*}} {
//       CHECKPARALLEL:         %[[IX:.*]] = affine.apply #[[$stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:         %[[IY:.*]] = affine.apply #[[$stride1Dilation1]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:         %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
//       CHECKPARALLEL:         %{{.*}} = load %{{.*}}[%[[IX]], %[[IY]]] : memref<?x?xf32>
//       CHECKPARALLEL:         %[[RES:.*]] = select %{{.*}}, %{{.*}}, %{{.*}} : f32
//       CHECKPARALLEL:         store %[[RES]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>

func @pooling_min(%arg0: memref<?x?xf32>,
                  %arg1: memref<?x?xi32>,
                  %arg2: memref<?x?xf32>) {
  linalg.pooling_min(%arg0, %arg1, %arg2) { strides = [2, 1] }:
    memref<?x?xf32>, memref<?x?xi32>, memref<?x?xf32>
  return
}
// CHECKLOOP-LABEL: func @pooling_min
//       CHECKLOOP:   %[[WX:.*]] = dim %arg1, %c0 : memref<?x?xi32>
//       CHECKLOOP:   %[[WY:.*]] = dim %arg1, %c1 : memref<?x?xi32>
//       CHECKLOOP:   %[[OX:.*]] = dim %arg2, %c0 : memref<?x?xf32>
//       CHECKLOOP:   %[[OY:.*]] = dim %arg2, %c1 : memref<?x?xf32>
//       CHECKLOOP:   scf.for %{{.*}} = %{{.*}} to %[[OX]] step %{{.*}} {
//       CHECKLOOP:     scf.for %{{.*}} = %{{.*}} to %[[OY]] step %{{.*}} {
//       CHECKLOOP:       scf.for %{{.*}} = %{{.*}} to %[[WX]] step %{{.*}} {
//       CHECKLOOP:         scf.for %{{.*}} = %{{.*}} to %[[WY]] step %{{.*}} {
//       CHECKLOOP:           %[[IX:.*]] = affine.apply #[[$stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKLOOP:           %[[IY:.*]] = affine.apply #[[$stride1Dilation1]](%{{.*}}, %{{.*}})
//       CHECKLOOP:           %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
//       CHECKLOOP:           %{{.*}} = load %{{.*}}[%[[IX]], %[[IY]]] : memref<?x?xf32>
//       CHECKLOOP:           %[[RES:.*]] = select %{{.*}}, %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:           store %[[RES]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>

// CHECKPARALLEL-LABEL: func @pooling_min
//       CHECKPARALLEL:   %[[WX:.*]] = dim %arg1, %c0 : memref<?x?xi32>
//       CHECKPARALLEL:   %[[WY:.*]] = dim %arg1, %c1 : memref<?x?xi32>
//       CHECKPARALLEL:   %[[OX:.*]] = dim %arg2, %c0 : memref<?x?xf32>
//       CHECKPARALLEL:   %[[OY:.*]] = dim %arg2, %c1 : memref<?x?xf32>
//       CHECKPARALLEL:   scf.parallel (%{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}) to (%[[OX]], %[[OY]]) step (%{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     scf.for %{{.*}} = %{{.*}} to %[[WX]] step %{{.*}} {
//       CHECKPARALLEL:       scf.for %{{.*}} = %{{.*}} to %[[WY]] step %{{.*}} {
//       CHECKPARALLEL:         %[[IX:.*]] = affine.apply #[[$stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:         %[[IY:.*]] = affine.apply #[[$stride1Dilation1]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:         %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
//       CHECKPARALLEL:         %{{.*}} = load %{{.*}}[%[[IX]], %[[IY]]] : memref<?x?xf32>
//       CHECKPARALLEL:         %[[RES:.*]] = select %{{.*}}, %{{.*}}, %{{.*}} : f32
//       CHECKPARALLEL:         store %[[RES]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>

func @pooling_sum(%arg0: memref<?x?xf32>,
                  %arg1: memref<?x?xi32>,
                  %arg2: memref<?x?xf32>) {
  linalg.pooling_sum(%arg0, %arg1, %arg2) { strides = [2, 1] }:
    memref<?x?xf32>, memref<?x?xi32>, memref<?x?xf32>
  return
}
// CHECKLOOP-LABEL: func @pooling_sum
//       CHECKLOOP:   %[[WX:.*]] = dim %arg1, %c0 : memref<?x?xi32>
//       CHECKLOOP:   %[[WY:.*]] = dim %arg1, %c1 : memref<?x?xi32>
//       CHECKLOOP:   %[[OX:.*]] = dim %arg2, %c0 : memref<?x?xf32>
//       CHECKLOOP:   %[[OY:.*]] = dim %arg2, %c1 : memref<?x?xf32>
//       CHECKLOOP:   scf.for %{{.*}} = %{{.*}} to %[[OX]] step %{{.*}} {
//       CHECKLOOP:     scf.for %{{.*}} = %{{.*}} to %[[OY]] step %{{.*}} {
//       CHECKLOOP:       scf.for %{{.*}} = %{{.*}} to %[[WX]] step %{{.*}} {
//       CHECKLOOP:         scf.for %{{.*}} = %{{.*}} to %[[WY]] step %{{.*}} {
//       CHECKLOOP:           %[[IX:.*]] = affine.apply #[[$stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKLOOP:           %[[IY:.*]] = affine.apply #[[$stride1Dilation1]](%{{.*}}, %{{.*}})
//       CHECKLOOP:           %[[RHS:.*]] = load %{{.*}}[%[[IX]], %[[IY]]] : memref<?x?xf32>
//       CHECKLOOP:           %[[LHS:.*]] = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
//       CHECKLOOP:           %[[RES:.*]] = addf %[[LHS]], %[[RHS]] : f32
//       CHECKLOOP:           store %[[RES]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>

// CHECKPARALLEL-LABEL: func @pooling_sum
//       CHECKPARALLEL:   %[[WX:.*]] = dim %arg1, %c0 : memref<?x?xi32>
//       CHECKPARALLEL:   %[[WY:.*]] = dim %arg1, %c1 : memref<?x?xi32>
//       CHECKPARALLEL:   %[[OX:.*]] = dim %arg2, %c0 : memref<?x?xf32>
//       CHECKPARALLEL:   %[[OY:.*]] = dim %arg2, %c1 : memref<?x?xf32>
//       CHECKPARALLEL:   scf.parallel (%{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}) to (%[[OX]], %[[OY]]) step (%{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     scf.for %{{.*}} = %{{.*}} to %[[WX]] step %{{.*}} {
//       CHECKPARALLEL:       scf.for %{{.*}} = %{{.*}} to %[[WY]] step %{{.*}} {
//       CHECKPARALLEL:         %[[IX:.*]] = affine.apply #[[$stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:         %[[IY:.*]] = affine.apply #[[$stride1Dilation1]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:         %[[RHS:.*]] = load %{{.*}}[%[[IX]], %[[IY]]] : memref<?x?xf32>
//       CHECKPARALLEL:         %[[LHS:.*]] = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
//       CHECKPARALLEL:         %[[RES:.*]] = addf %[[LHS]], %[[RHS]] : f32
//       CHECKPARALLEL:         store %[[RES]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>

#accesses = [
  affine_map<(i, j, k) -> (i, j)>,
  affine_map<(i, j, k) -> (i, j, k)>,
  affine_map<(i, j, k) -> (i, k, j)>
]
#trait2 = {
  args_in = 1,
  args_out = 2,
  iterator_types = ["parallel", "parallel", "parallel"],
  indexing_maps = #accesses,
  library_call = "some_external_function_name_2",
  doc = "B(i,j,k), C(i,k,j) = foo(A(i, j), B(i,j,k), C(i,k,j))"
}
func @generic_region(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.generic #trait2 %arg0, %arg1, %arg2 {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %d = mulf %a, %b : f32
      %e = addf %c, %d : f32
      linalg.yield %d, %e : f32, f32
  }: memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECKLOOP-LABEL: @generic_region
//       CHECKLOOP: scf.for %[[i:.*]] = {{.*}}
//       CHECKLOOP:   scf.for %[[j:.*]] = {{.*}}
//       CHECKLOOP:     scf.for %[[k:.*]] = {{.*}}
//       CHECKLOOP:       %[[a:.*]] = load %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32, #[[$strided2D]]>
//       CHECKLOOP:       %[[b:.*]] = load %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKLOOP:       %[[c:.*]] = load %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKLOOP:       %[[d:.*]] = mulf %[[a]], %[[b]] : f32
//       CHECKLOOP:       %[[e:.*]] = addf %[[c]], %[[d]] : f32
//       CHECKLOOP:       store %[[d]], %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKLOOP:       store %[[e]], %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[$strided3D]]>

// CHECKPARALLEL-LABEL: @generic_region
//       CHECKPARALLEL: scf.parallel (%[[i:[a-zA-Z0-9_]*]], %[[j:[a-zA-Z0-9_]*]], %[[k:[a-zA-Z0-9_]*]])
//       CHECKPARALLEL:   %[[a:.*]] = load %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32, #[[$strided2D]]>
//       CHECKPARALLEL:   %[[b:.*]] = load %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:   %[[c:.*]] = load %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:   %[[d:.*]] = mulf %[[a]], %[[b]] : f32
//       CHECKPARALLEL:   %[[e:.*]] = addf %[[c]], %[[d]] : f32
//       CHECKPARALLEL:   store %[[d]], %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:   store %[[e]], %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[$strided3D]]>

#trait4 = {
  args_in = 1,
  args_out = 2,
  iterator_types = ["parallel", "parallel", "parallel"],
  indexing_maps = #accesses,
  library_call = "some_external_function_name_2",
  doc = "B(i,j,k), C(i,k,j) = foo(A(i, j) * B(i,j,k), i * j * k + C(i,k,j))"
}
func @indexed_generic_region(
        %arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>,
        %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>,
        %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.indexed_generic #trait4 %arg0, %arg1, %arg2 {
    ^bb0(%i: index, %j: index, %k: index, %a: f32, %b: f32, %c: f32):
      %result_1 = mulf %a, %b : f32

      %ij = addi %i, %j : index
      %ijk = addi %ij, %k : index
      %ijk_int = index_cast %ijk : index to i32
      %ijk_float = sitofp %ijk_int : i32 to f32

      %result_2 = addf %c, %ijk_float : f32
      linalg.yield %result_1, %result_2 : f32, f32
  }: memref<?x?xf32, offset: ?, strides: [?, 1]>,
     memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>,
     memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}

// CHECKLOOP-LABEL: @indexed_generic_region
//       CHECKLOOP: scf.for %[[i:.*]] = {{.*}}
//       CHECKLOOP:   scf.for %[[j:.*]] = {{.*}}
//       CHECKLOOP:     scf.for %[[k:.*]] = {{.*}}
//       CHECKLOOP:       %[[a:.*]] = load %{{.*}}[%[[i]], %[[j]]]
//       CHECKLOOP:       %[[b:.*]] = load %{{.*}}[%[[i]], %[[j]], %[[k]]]
//       CHECKLOOP:       %[[c:.*]] = load %{{.*}}[%[[i]], %[[k]], %[[j]]]
//       CHECKLOOP:       %[[result_1:.*]] = mulf %[[a]], %[[b]] : f32
//       CHECKLOOP:       %[[ij:.*]] = addi %[[i]], %[[j]] : index
//       CHECKLOOP:       %[[ijk:.*]] = addi %[[ij]], %[[k]] : index
//       CHECKLOOP:       %[[ijk_int:.*]] = index_cast %[[ijk]] : index to i32
//       CHECKLOOP:       %[[ijk_float:.*]] = sitofp %[[ijk_int]] : i32 to f32
//       CHECKLOOP:       %[[result_2:.*]] = addf %[[c]], %[[ijk_float]] : f32
//       CHECKLOOP:       store %[[result_1]], %{{.*}}[%[[i]], %[[j]], %[[k]]]
//       CHECKLOOP:       store %[[result_2]], %{{.*}}[%[[i]], %[[k]], %[[j]]]

// CHECKPARALLEL-LABEL: @indexed_generic_region
//       CHECKPARALLEL: scf.parallel (%[[i:[a-zA-Z0-9_]*]], %[[j:[a-zA-Z0-9_]*]], %[[k:[a-zA-Z0-9_]*]])
//       CHECKPARALLEL:   %[[a:.*]] = load %{{.*}}[%[[i]], %[[j]]]
//       CHECKPARALLEL:   %[[b:.*]] = load %{{.*}}[%[[i]], %[[j]], %[[k]]]
//       CHECKPARALLEL:   %[[c:.*]] = load %{{.*}}[%[[i]], %[[k]], %[[j]]]
//       CHECKPARALLEL:   %[[result_1:.*]] = mulf %[[a]], %[[b]] : f32
//       CHECKPARALLEL:   %[[ij:.*]] = addi %[[i]], %[[j]] : index
//       CHECKPARALLEL:   %[[ijk:.*]] = addi %[[ij]], %[[k]] : index
//       CHECKPARALLEL:   %[[ijk_int:.*]] = index_cast %[[ijk]] : index to i32
//       CHECKPARALLEL:   %[[ijk_float:.*]] = sitofp %[[ijk_int]] : i32 to f32
//       CHECKPARALLEL:   %[[result_2:.*]] = addf %[[c]], %[[ijk_float]] : f32
//       CHECKPARALLEL:   store %[[result_1]], %{{.*}}[%[[i]], %[[j]], %[[k]]]
//       CHECKPARALLEL:   store %[[result_2]], %{{.*}}[%[[i]], %[[k]], %[[j]]]

// -----

#broadcast_access = [
  affine_map<(i, j) -> ()>,
  affine_map<(i, j) -> (i, j)>
]

#trait_broadcast = {
  args_in = 1,
  args_out = 1,
  indexing_maps = #broadcast_access,
  iterator_types = ["parallel", "parallel"],
  library_call = "some_broadcast_external_fn"
}

func @generic_op_zero_rank(%arg0: memref<f32>, %arg1: memref<3x4xf32>)
{
  linalg.generic #trait_broadcast %arg0, %arg1 {
    ^bb(%a: f32, %b: f32) :
      linalg.yield %a : f32
  } : memref<f32>, memref<3x4xf32>
  return
}

// CHECKLOOP-LABEL: @generic_op_zero_rank
//  CHECKLOOP-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECKLOOP-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<3x4xf32>
//       CHECKLOOP: scf.for %[[i:.*]] = {{.*}}
//       CHECKLOOP:   scf.for %[[j:.*]] = {{.*}}
//       CHECKLOOP:     %[[a:.*]] = load %[[ARG0]][]
//       CHECKLOOP:     store %[[a]], %[[ARG1]][%[[i]], %[[j]]]

// CHECKPARALLEL-LABEL: @generic_op_zero_rank
//  CHECKPARALLEL-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECKPARALLEL-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<3x4xf32>
//       CHECKPARALLEL: scf.parallel (%[[i:[a-zA-Z0-9_]*]], %[[j:[a-zA-Z0-9_]*]])
//       CHECKPARALLEL:   %[[a:.*]] = load %[[ARG0]][]
//       CHECKPARALLEL:   store %[[a]], %[[ARG1]][%[[i]], %[[j]]]

func @indexed_generic_op_zero_rank(%arg0: memref<i32>, %arg1: memref<3x4xi32>)
{
  linalg.indexed_generic #trait_broadcast %arg0, %arg1 {
    ^bb(%i: index, %j: index, %a: i32, %b: i32) :
      %ij = addi %i, %j : index
      %ij_int = index_cast %ij : index to i32
      %result = addi %a, %ij_int : i32
      linalg.yield %result : i32
  } : memref<i32>, memref<3x4xi32>
  return
}

// CHECKLOOP-LABEL: @indexed_generic_op_zero_rank
//  CHECKLOOP-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<i32>
//  CHECKLOOP-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<3x4xi32>
//       CHECKLOOP: scf.for %[[i:.*]] = {{.*}}
//       CHECKLOOP:   scf.for %[[j:.*]] = {{.*}}
//       CHECKLOOP:     %[[a:.*]] = load %[[ARG0]][
//       CHECKLOOP:     %[[ij:.*]] = addi %[[i]], %[[j]] : index
//       CHECKLOOP:     %[[ij_int:.*]] = index_cast %[[ij]] : index to i32
//       CHECKLOOP:     %[[result:.*]] = addi %[[a]], %[[ij_int]] : i32
//       CHECKLOOP:     store %[[result]], %[[ARG1]][%[[i]], %[[j]]]

// CHECKPARALLEL-LABEL: @indexed_generic_op_zero_rank
//  CHECKPARALLEL-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<i32>
//  CHECKPARALLEL-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<3x4xi32>
//       CHECKPARALLEL: scf.parallel (%[[i:[a-zA-Z0-9_]*]], %[[j:[a-zA-Z0-9_]*]])
//       CHECKPARALLEL:   %[[a:.*]] = load %[[ARG0]][
//       CHECKPARALLEL:   %[[ij:.*]] = addi %[[i]], %[[j]] : index
//       CHECKPARALLEL:   %[[ij_int:.*]] = index_cast %[[ij]] : index to i32
//       CHECKPARALLEL:   %[[result:.*]] = addi %[[a]], %[[ij_int]] : i32
//       CHECKPARALLEL:   store %[[result]], %[[ARG1]][%[[i]], %[[j]]]

#reduce_1D_access = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> ()>
]

#trait_reduce_1D = {
  args_in = 1,
  args_out = 1,
  indexing_maps = #reduce_1D_access,
  iterator_types = ["reduction"],
  library_call = "some_reduce_external_fn"
}

func @generic_op_1D_reduce(%arg0: memref<?xf32>, %arg1: memref<f32>)
{
  linalg.generic #trait_reduce_1D %arg0, %arg1 {
    ^bb(%a: f32, %b: f32) :
      %0 = addf %a, %b : f32
      linalg.yield %0 : f32
  } : memref<?xf32>, memref<f32>
  return
}
// CHECKLOOP-LABEL: @generic_op_1D_reduce
//  CHECKLOOP-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?xf32>
//  CHECKLOOP-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
//       CHECKLOOP: scf.for %[[i:.*]] = {{.*}}
//       CHECKLOOP:   %[[a:.*]] = load %[[ARG0]][%[[i]]]
//       CHECKLOOP:   %[[b:.*]] = load %[[ARG1]][]
//       CHECKLOOP:   %[[c:.*]] = addf %[[a]], %[[b]] : f32
//       CHECKLOOP:   store %[[c]], %[[ARG1]][]

// CHECKPARALLEL-LABEL: @generic_op_1D_reduce
//  CHECKPARALLEL-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?xf32>
//  CHECKPARALLEL-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
//       CHECKPARALLEL: scf.for %[[i:.*]] = {{.*}}
//       CHECKPARALLEL:   %[[a:.*]] = load %[[ARG0]][%[[i]]]
//       CHECKPARALLEL:   %[[b:.*]] = load %[[ARG1]][]
//       CHECKPARALLEL:   %[[c:.*]] = addf %[[a]], %[[b]] : f32
//       CHECKPARALLEL:   store %[[c]], %[[ARG1]][]


#reduce_init_1D_access = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> ()>,
  affine_map<(i) -> ()>
]

#trait_reduce_init_1D = {
  args_in = 2,
  args_out = 1,
  indexing_maps = #reduce_init_1D_access,
  iterator_types = ["reduction"],
  library_call = "some_reduce_external_fn"
}

func @indexed_generic_op_1D_reduce(%arg0: memref<?xf32>,
                                   %arg1: memref<f32>,
                                   %arg2: memref<f32>)
{
  linalg.indexed_generic #trait_reduce_init_1D %arg0, %arg1, %arg2 {
    ^bb(%i : index, %a: f32, %b: f32, %c: f32) :
      %0 = constant 0 : index
      %1 = cmpi "eq", %0, %i : index
      %2 = select %1, %b, %c : f32
      %3 = addf %a, %2 : f32
      linalg.yield %3 : f32
  } : memref<?xf32>, memref<f32>, memref<f32>
  return
}
// CHECKLOOP-LABEL: @indexed_generic_op_1D_reduce
//  CHECKLOOP-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?xf32>
//  CHECKLOOP-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECKLOOP-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<f32>
//       CHECKLOOP: scf.for %[[i:.*]] = {{.*}}
//       CHECKLOOP:   %[[a:.*]] = load %[[ARG0]][%[[i]]]
//       CHECKLOOP:   %[[b:.*]] = load %[[ARG1]][]
//       CHECKLOOP:   %[[c:.*]] = load %[[ARG2]][]
//       CHECKLOOP:   %[[d:.*]] = select %{{.*}}, %[[b]], %[[c]]
//       CHECKLOOP:   %[[e:.*]] = addf %[[a]], %[[d]]
//       CHECKLOOP:   store %[[e]], %[[ARG2]][]

// CHECKPARALLEL-LABEL: @indexed_generic_op_1D_reduce
//  CHECKPARALLEL-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?xf32>
//  CHECKPARALLEL-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECKPARALLEL-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<f32>
//       CHECKPARALLEL: scf.for %[[i:.*]] = {{.*}}
//       CHECKPARALLEL:   %[[a:.*]] = load %[[ARG0]][%[[i]]]
//       CHECKPARALLEL:   %[[b:.*]] = load %[[ARG1]][]
//       CHECKPARALLEL:   %[[c:.*]] = load %[[ARG2]][]
//       CHECKPARALLEL:   %[[d:.*]] = select %{{.*}}, %[[b]], %[[c]]
//       CHECKPARALLEL:   %[[e:.*]] = addf %[[a]], %[[d]]
//       CHECKPARALLEL:   store %[[e]], %[[ARG2]][]

#trait_const_fill = {
  args_in = 0,
  args_out = 1,
  indexing_maps = [affine_map<(i) -> (i)>],
  iterator_types = ["parallel"],
  library_call = "some_external_fn"
}
func @generic_const_init(%arg0: memref<?xf32>) {
        %cst = constant 1.0 : f32
  linalg.generic #trait_const_fill %arg0 {
    ^bb0(%arg1: f32):   // no predecessors
      linalg.yield %cst : f32
    }: memref<?xf32>
    return
}
// CHECKLOOP-LABEL: @generic_const_init
//  CHECKLOOP-SAME: %[[ARG0:.*]]: memref<?xf32>
//       CHECKLOOP: %[[CONST:.*]] = constant 1.000000e+00 : f32
//       CHECKLOOP: scf.for %[[i:.*]] = {{.*}}
//       CHECKLOOP:   store %[[CONST]], %[[ARG0]]

// CHECKPARALLEL-LABEL: @generic_const_init
//  CHECKPARALLEL-SAME: %[[ARG0:.*]]: memref<?xf32>
//       CHECKPARALLEL: %[[CONST:.*]] = constant 1.000000e+00 : f32
//       CHECKPARALLEL: scf.parallel (%[[i:.*]])
//       CHECKPARALLEL:   store %[[CONST]], %[[ARG0]]

#scalar_access = [
  affine_map<() -> ()>,
  affine_map<() -> ()>,
  affine_map<() -> ()>
]
#scalar_trait = {
  args_in = 2,
  args_out = 1,
  iterator_types = [],
  indexing_maps = #scalar_access,
  library_call = "some_external_fn"
}
func @scalar_code(%arg0: memref<f32>, %arg1 : memref<f32>, %arg2 : memref<f32>)
{
  linalg.generic #scalar_trait %arg0, %arg1, %arg2 {
  ^bb(%a : f32, %b : f32, %c : f32) :
    %0 = addf %a, %b : f32
    linalg.yield %0 : f32
  } : memref<f32>, memref<f32>, memref<f32>
  return
}
// CHECKLOOP-LABEL: @scalar_code
//  CHECKLOOP-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECKLOOP-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECKLOOP-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<f32>
//   CHECKLOOP-NOT: scf.for
//       CHECKLOOP: load %[[ARG0]][]
//       CHECKLOOP: load %[[ARG1]][]
//       CHECKLOOP: addf
//       CHECKLOOP: store %{{.*}}, %[[ARG2]][]

// CHECKPARALLEL-LABEL: @scalar_code
//  CHECKPARALLEL-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECKPARALLEL-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECKPARALLEL-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<f32>
//   CHECKPARALLEL-NOT: scf.for
//       CHECKPARALLEL: load %[[ARG0]][]
//       CHECKPARALLEL: load %[[ARG1]][]
//       CHECKPARALLEL: addf
//       CHECKPARALLEL: store %{{.*}}, %[[ARG2]][]

//----------------------------------------------------------------------------//
// Named ops to loops.
//----------------------------------------------------------------------------//
func @named_batch_matmul(%A: memref<?x?x?xf32>, %B: memref<?x?x?xf32>, %C: memref<?x?x?xf32>) {
  linalg.batch_matmul %A, %B, %C : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
  return
}
// CHECKLOOP-LABEL: @named_batch_matmul
//  CHECKLOOP-SAME: %[[mA:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECKLOOP-SAME: %[[mB:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECKLOOP-SAME: %[[mC:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//       CHECKLOOP: %[[B:.*]] = dim %[[mA]], %c0 : memref<?x?x?xf32>
//       CHECKLOOP: %[[M:.*]] = dim %[[mA]], %c1 : memref<?x?x?xf32>
//       CHECKLOOP: %[[K:.*]] = dim %[[mA]], %c2 : memref<?x?x?xf32>
//       CHECKLOOP: %[[N:.*]] = dim %[[mB]], %c2 : memref<?x?x?xf32>
//       CHECKLOOP: scf.for %[[b:.*]] = %{{.*}} to %[[B]] step %{{.*}} {
//       CHECKLOOP:   scf.for %[[m:.*]] = %{{.*}} to %[[M]] step %{{.*}} {
//       CHECKLOOP:     scf.for %[[n:.*]] = %{{.*}} to %[[N]] step %{{.*}} {
//       CHECKLOOP:       scf.for %[[k:.*]] = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECKLOOP:       %[[va:.*]] = load %[[mA]][%[[b]], %[[m]], %[[k]]] : memref<?x?x?xf32>
//       CHECKLOOP:       %[[vb:.*]] = load %[[mB]][%[[b]], %[[k]], %[[n]]] : memref<?x?x?xf32>
//       CHECKLOOP:       %[[vc:.*]] = load %[[mC]][%[[b]], %[[m]], %[[n]]] : memref<?x?x?xf32>
//       CHECKLOOP:       %[[inc:.*]] = mulf %[[va]], %[[vb]] : f32
//       CHECKLOOP:       %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECKLOOP:       store %[[res]], %[[mC]][%[[b]], %[[m]], %[[n]]] : memref<?x?x?xf32>

// CHECKPARALLEL-LABEL: @named_batch_matmul
//  CHECKPARALLEL-SAME: %[[mA:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECKPARALLEL-SAME: %[[mB:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECKPARALLEL-SAME: %[[mC:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//       CHECKPARALLEL: %[[B:.*]] = dim %[[mA]], %c0 : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[M:.*]] = dim %[[mA]], %c1 : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[K:.*]] = dim %[[mA]], %c2 : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[N:.*]] = dim %[[mB]], %c2 : memref<?x?x?xf32>
//       CHECKPARALLEL: scf.parallel (%[[b:.*]], %[[m:.*]], %[[n:.*]]) = ({{.*}}) to (%[[B]], %[[M]], %[[N]]) step ({{.*}}) {
//       CHECKPARALLEL:   scf.for %[[k:.*]] = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECKPARALLEL:       %[[va:.*]] = load %[[mA]][%[[b]], %[[m]], %[[k]]] : memref<?x?x?xf32>
//       CHECKPARALLEL:       %[[vb:.*]] = load %[[mB]][%[[b]], %[[k]], %[[n]]] : memref<?x?x?xf32>
//       CHECKPARALLEL:       %[[vc:.*]] = load %[[mC]][%[[b]], %[[m]], %[[n]]] : memref<?x?x?xf32>
//       CHECKPARALLEL:       %[[inc:.*]] = mulf %[[va]], %[[vb]] : f32
//       CHECKPARALLEL:       %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECKPARALLEL:       store %[[res]], %[[mC]][%[[b]], %[[m]], %[[n]]] : memref<?x?x?xf32>

#conv_1d_accesses = [
  affine_map<(m, n)[s0] -> (m + n - s0 floordiv 2)>, // in
  affine_map<(m, n)[s0] -> (n)>, // filter
  affine_map<(m, n)[s0] -> (m)> // out
]

#conv_1d_trait = {
  args_in = 2,
  args_out = 1,
  doc = "C(m) += A(m) * B(n)",
  indexing_maps = #conv_1d_accesses,
  library_call  = "linalg_conv_1d",
  n_views = [2, 1],
  iterator_types = ["parallel", "parallel"],
  symbol_source = 1
}

func @conv1d(%in : memref<?xf32>, %filter : memref<?xf32>, %out :  memref<?xf32>) -> () {
  linalg.generic #conv_1d_trait %in, %filter, %out {
    ^bb0(%a: f32, %b: f32, %c: f32) :
      %d = mulf %a, %b : f32
      %e = addf %c, %d : f32
      linalg.yield %e : f32
  } : memref<?xf32>,
      memref<?xf32>,
      memref<?xf32>
  return
}

// CHECKLOOP-LABEL: @conv1d
//  CHECKLOOP-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?xf32>
//  CHECKLOOP-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?xf32>
//  CHECKLOOP-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?xf32>
//       CHECKLOOP: %[[c0:.*]] = constant 0 : index
//       CHECKLOOP: %[[dim0:.*]] = dim %[[arg1]], %[[c0]] : memref<?xf32>
//       CHECKLOOP: %[[dim1:.*]] = dim %[[arg2]], %[[c0]] : memref<?xf32>
//       CHECKLOOP: scf.for %[[b:.*]] = %{{.*}} to %[[dim1]] step %{{.*}} {
//       CHECKLOOP:   scf.for %[[m:.*]] = %{{.*}} to %[[dim0]] step %{{.*}} {
//       CHECKLOOP:     %[[dim2:.*]] = dim %[[arg1]], %[[c0]] : memref<?xf32>
//       CHECKLOOP:     %[[aff:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim2]]]
//       CHECKLOOP:     %[[va:.*]] = load %[[arg0]][%[[aff]]] : memref<?xf32>
//       CHECKLOOP:     %[[vb:.*]] = load %[[arg1]][%[[m]]] : memref<?xf32>
//       CHECKLOOP:     %[[vc:.*]] = load %[[arg2]][%[[b]]] : memref<?xf32>
//       CHECKLOOP:     %[[inc:.*]] = mulf %[[va]], %[[vb]] : f32
//       CHECKLOOP:     %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECKLOOP:     store %[[res]], %[[arg2]][%[[b]]] : memref<?xf32>

// CHECKPARALLEL-LABEL: @conv1d
//  CHECKPARALLEL-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?xf32>
//  CHECKPARALLEL-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?xf32>
//  CHECKPARALLEL-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?xf32>
//       CHECKPARALLEL: %[[c0:.*]] = constant 0 : index
//       CHECKPARALLEL: %[[dim0:.*]] = dim %[[arg1]], %[[c0]] : memref<?xf32>
//       CHECKPARALLEL: %[[dim1:.*]] = dim %[[arg2]], %[[c0]] : memref<?xf32>
//       CHECKPARALLEL: scf.parallel (%[[b:.*]], %[[m:.*]]) = (%{{.*}}, %{{.*}}) to (%[[dim1]], %[[dim0]]) step ({{.*}}) {
//       CHECKPARALLEL:   %[[dim2:.*]] = dim %[[arg1]], %[[c0]] : memref<?xf32>
//       CHECKPARALLEL:   %[[aff:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim2]]]
//       CHECKPARALLEL:   %[[va:.*]] = load %[[arg0]][%[[aff]]] : memref<?xf32>
//       CHECKPARALLEL:   %[[vb:.*]] = load %[[arg1]][%[[m]]] : memref<?xf32>
//       CHECKPARALLEL:   %[[vc:.*]] = load %[[arg2]][%[[b]]] : memref<?xf32>
//       CHECKPARALLEL:   %[[inc:.*]] = mulf %[[va]], %[[vb]] : f32
//       CHECKPARALLEL:   %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECKPARALLEL:   store %[[res]], %[[arg2]][%[[b]]] : memref<?xf32>

#conv_2d_accesses = [
  affine_map<(m, n, m1, n1)[s0, s1] -> (m + m1 - s0 floordiv 2, n + n1 - s1 floordiv 2)>, // in
  affine_map<(m, n, m1, n1)[s0, s1] -> (m1, n1)>, // filter
  affine_map<(m, n, m1, n1)[s0, s1] -> (m, n)> // out
]

#conv_2d_trait = {
  args_in = 2,
  args_out = 1,
  doc = "C(m,n) += A(m,n) * B(m1,n1)",
  indexing_maps = #conv_2d_accesses,
  library_call  = "linalg_conv_2d",
  n_views = [2, 1],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"],
  symbol_source = 1
}

func @conv2d(%in : memref<?x?xf32>, %filter : memref<?x?xf32>, %out :  memref<?x?xf32>) -> () {
  linalg.generic #conv_2d_trait %in, %filter, %out {
    ^bb0(%a: f32, %b: f32, %c: f32) :
      %d = mulf %a, %b : f32
      %e = addf %c, %d : f32
      linalg.yield %e : f32
  } : memref<?x?xf32>,
      memref<?x?xf32>,
      memref<?x?xf32>
  return
}

// CHECKLOOP-LABEL: @conv2d
//  CHECKLOOP-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?x?xf32>
//  CHECKLOOP-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?x?xf32>
//  CHECKLOOP-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?xf32>
//       CHECKLOOP: %[[c0:.*]] = constant 0 : index
//       CHECKLOOP: %[[c1:.*]] = constant 1 : index
//       CHECKLOOP: %[[dim0:.*]] = dim %[[arg1]], %[[c0]] : memref<?x?xf32>
//       CHECKLOOP: %[[dim1:.*]] = dim %[[arg1]], %[[c1]] : memref<?x?xf32>
//       CHECKLOOP: %[[dim2:.*]] = dim %[[arg2]], %[[c0]] : memref<?x?xf32>
//       CHECKLOOP: %[[dim3:.*]] = dim %[[arg2]], %[[c1]] : memref<?x?xf32>
//       CHECKLOOP: scf.for %[[i0:.*]] = %{{.*}} to %[[dim2]] step %{{.*}} {
//       CHECKLOOP:   scf.for %[[i1:.*]] = %{{.*}} to %[[dim3]] step %{{.*}} {
//       CHECKLOOP:     scf.for %[[i2:.*]] = %{{.*}} to %[[dim0]] step %{{.*}} {
//       CHECKLOOP:       scf.for %[[i3:.*]] = %{{.*}} to %[[dim1]] step %{{.*}} {
//       CHECKLOOP:         %[[dim4:.*]] = dim %[[arg1]], %[[c0]] : memref<?x?xf32>
//       CHECKLOOP:         %[[dim5:.*]] = dim %[[arg1]], %[[c1]] : memref<?x?xf32>
//       CHECKLOOP:         %[[aff1:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim4]]]
//       CHECKLOOP:         %[[aff2:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim5]]]
//       CHECKLOOP:         %[[va:.*]] = load %[[arg0]][%[[aff1]], %[[aff2]]] : memref<?x?xf32>
//       CHECKLOOP:         %[[vb:.*]] = load %[[arg1]][%[[i2]], %[[i3]]] : memref<?x?xf32>
//       CHECKLOOP:         %[[vc:.*]] = load %[[arg2]][%[[i0]], %[[i1]]] : memref<?x?xf32>
//       CHECKLOOP:         %[[inc:.*]] = mulf %[[va]], %[[vb]] : f32
//       CHECKLOOP:         %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECKLOOP:         store %[[res]], %[[arg2]][%[[i0]], %[[i1]]] : memref<?x?xf32>

// CHECKPARALLEL-LABEL: @conv2d
//  CHECKPARALLEL-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?x?xf32>
//  CHECKPARALLEL-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?x?xf32>
//  CHECKPARALLEL-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?xf32>
//       CHECKPARALLEL: %[[c0:.*]] = constant 0 : index
//       CHECKPARALLEL: %[[c1:.*]] = constant 1 : index
//       CHECKPARALLEL: %[[dim0:.*]] = dim %[[arg1]], %[[c0]] : memref<?x?xf32>
//       CHECKPARALLEL: %[[dim1:.*]] = dim %[[arg1]], %[[c1]] : memref<?x?xf32>
//       CHECKPARALLEL: %[[dim2:.*]] = dim %[[arg2]], %[[c0]] : memref<?x?xf32>
//       CHECKPARALLEL: %[[dim3:.*]] = dim %[[arg2]], %[[c1]] : memref<?x?xf32>
//       CHECKPARALLEL: scf.parallel (%[[i0:.*]], %[[i1:.*]], %[[i2:.*]], %[[i3:.*]]) = (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) to (%[[dim2]], %[[dim3]], %[[dim0]], %[[dim1]]) step ({{.*}}) {
//       CHECKPARALLEL:   %[[dim4:.*]] = dim %[[arg1]], %[[c0]] : memref<?x?xf32>
//       CHECKPARALLEL:   %[[dim5:.*]] = dim %[[arg1]], %[[c1]] : memref<?x?xf32>
//       CHECKPARALLEL:   %[[aff1:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim4]]]
//       CHECKPARALLEL:   %[[aff2:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim5]]]
//       CHECKPARALLEL:   %[[va:.*]] = load %[[arg0]][%[[aff1]], %[[aff2]]] : memref<?x?xf32>
//       CHECKPARALLEL:   %[[vb:.*]] = load %[[arg1]][%[[i2]], %[[i3]]] : memref<?x?xf32>
//       CHECKPARALLEL:   %[[vc:.*]] = load %[[arg2]][%[[i0]], %[[i1]]] : memref<?x?xf32>
//       CHECKPARALLEL:   %[[inc:.*]] = mulf %[[va]], %[[vb]] : f32
//       CHECKPARALLEL:   %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECKPARALLEL:   store %[[res]], %[[arg2]][%[[i0]], %[[i1]]] : memref<?x?xf32>

#conv_3d_accesses = [
  affine_map<(m, n, k, m1, n1, k1)[s0, s1, s2] -> (m + m1 - s0 floordiv 2, n + n1 - s1 floordiv 2, k + k1 - s2 floordiv 2)>, // in
  affine_map<(m, n, k, m1, n1, k1)[s0, s1, s2] -> (m1, n1, k1)>, // filter
  affine_map<(m, n, k, m1, n1, k1)[s0, s1, s2] -> (m, n, k)> // out
]

#conv_3d_trait = {
  args_in = 2,
  args_out = 1,
  doc = "C(m,n,k) += A(m,n,k) * B(m1,n1,k1)",
  indexing_maps = #conv_3d_accesses,
  library_call  = "linalg_conv_3d",
  n_views = [2, 1],
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"],
  symbol_source = 1
}

func @conv3d(%in : memref<?x?x?xf32>, %filter : memref<?x?x?xf32>, %out :  memref<?x?x?xf32>) -> () {
  linalg.generic #conv_3d_trait %in, %filter, %out {
    ^bb0(%a: f32, %b: f32, %c: f32) :
      %d = mulf %a, %b : f32
      %e = addf %c, %d : f32
      linalg.yield %e : f32
  } : memref<?x?x?xf32>,
      memref<?x?x?xf32>,
      memref<?x?x?xf32>
  return
}

// CHECKLOOP-LABEL: @conv3d
//  CHECKLOOP-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECKLOOP-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECKLOOP-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//       CHECKLOOP: %[[c0:.*]] = constant 0 : index
//       CHECKLOOP: %[[c1:.*]] = constant 1 : index
//       CHECKLOOP: %[[c2:.*]] = constant 2 : index
//       CHECKLOOP: %[[dim0:.*]] = dim %[[arg1]], %[[c0]] : memref<?x?x?xf32>
//       CHECKLOOP: %[[dim1:.*]] = dim %[[arg1]], %[[c1]] : memref<?x?x?xf32>
//       CHECKLOOP: %[[dim2:.*]] = dim %[[arg1]], %[[c2]] : memref<?x?x?xf32>
//       CHECKLOOP: %[[dim3:.*]] = dim %[[arg2]], %[[c0]] : memref<?x?x?xf32>
//       CHECKLOOP: %[[dim4:.*]] = dim %[[arg2]], %[[c1]] : memref<?x?x?xf32>
//       CHECKLOOP: %[[dim5:.*]] = dim %[[arg2]], %[[c2]] : memref<?x?x?xf32>
//       CHECKLOOP: scf.for %[[i0:.*]] = %{{.*}} to %[[dim3]] step %{{.*}} {
//       CHECKLOOP:   scf.for %[[i1:.*]] = %{{.*}} to %[[dim4]] step %{{.*}} {
//       CHECKLOOP:     scf.for %[[i2:.*]] = %{{.*}} to %[[dim5]] step %{{.*}} {
//       CHECKLOOP:       scf.for %[[i3:.*]] = %{{.*}} to %[[dim0]] step %{{.*}} {
//       CHECKLOOP:         scf.for %[[i4:.*]] = %{{.*}} to %[[dim1]] step %{{.*}} {
//       CHECKLOOP:           scf.for %[[i5:.*]] = %{{.*}} to %[[dim2]] step %{{.*}} {
//       CHECKLOOP:             %[[dim6:.*]] = dim %[[arg1]], %[[c0]] : memref<?x?x?xf32>
//       CHECKLOOP:             %[[dim7:.*]] = dim %[[arg1]], %[[c1]] : memref<?x?x?xf32>
//       CHECKLOOP:             %[[dim8:.*]] = dim %[[arg1]], %[[c2]] : memref<?x?x?xf32>
//       CHECKLOOP:             %[[aff1:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim6]]]
//       CHECKLOOP:             %[[aff2:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim7]]]
//       CHECKLOOP:             %[[aff3:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim8]]]
//       CHECKLOOP:             %[[va:.*]] = load %[[arg0]][%[[aff1]], %[[aff2]], %[[aff3]]] : memref<?x?x?xf32>
//       CHECKLOOP:             %[[vb:.*]] = load %[[arg1]][%[[i3]], %[[i4]], %[[i5]]] : memref<?x?x?xf32>
//       CHECKLOOP:             %[[vc:.*]] = load %[[arg2]][%[[i0]], %[[i1]], %[[i2]]] : memref<?x?x?xf32>
//       CHECKLOOP:             %[[inc:.*]] = mulf %[[va]], %[[vb]] : f32
//       CHECKLOOP:             %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECKLOOP:             store %[[res]], %[[arg2]][%[[i0]], %[[i1]], %[[i2]]] : memref<?x?x?xf32>

// CHECKPARALLEL-LABEL: @conv3d
//  CHECKPARALLEL-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECKPARALLEL-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECKPARALLEL-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//       CHECKPARALLEL: %[[c0:.*]] = constant 0 : index
//       CHECKPARALLEL: %[[c1:.*]] = constant 1 : index
//       CHECKPARALLEL: %[[c2:.*]] = constant 2 : index
//       CHECKPARALLEL: %[[dim0:.*]] = dim %[[arg1]], %[[c0]] : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[dim1:.*]] = dim %[[arg1]], %[[c1]] : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[dim2:.*]] = dim %[[arg1]], %[[c2]] : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[dim3:.*]] = dim %[[arg2]], %[[c0]] : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[dim4:.*]] = dim %[[arg2]], %[[c1]] : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[dim5:.*]] = dim %[[arg2]], %[[c2]] : memref<?x?x?xf32>
//       CHECKPARALLEL: scf.parallel (%[[i0:.*]], %[[i1:.*]], %[[i2:.*]], %[[i3:.*]], %[[i4:.*]], %[[i5:.*]]) = (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) to (%[[dim3]], %[[dim4]], %[[dim5]], %[[dim0]], %[[dim1]], %[[dim2]]) step ({{.*}}) {
//       CHECKPARALLEL:   %[[dim6:.*]] = dim %[[arg1]], %[[c0]] : memref<?x?x?xf32>
//       CHECKPARALLEL:   %[[dim7:.*]] = dim %[[arg1]], %[[c1]] : memref<?x?x?xf32>
//       CHECKPARALLEL:   %[[dim8:.*]] = dim %[[arg1]], %[[c2]] : memref<?x?x?xf32>
//       CHECKPARALLEL:   %[[aff1:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim6]]]
//       CHECKPARALLEL:   %[[aff2:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim7]]]
//       CHECKPARALLEL:   %[[aff3:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim8]]]
//       CHECKPARALLEL:   %[[va:.*]] = load %[[arg0]][%[[aff1]], %[[aff2]], %[[aff3]]] : memref<?x?x?xf32>
//       CHECKPARALLEL:   %[[vb:.*]] = load %[[arg1]][%[[i3]], %[[i4]], %[[i5]]] : memref<?x?x?xf32>
//       CHECKPARALLEL:   %[[vc:.*]] = load %[[arg2]][%[[i0]], %[[i1]], %[[i2]]] : memref<?x?x?xf32>
//       CHECKPARALLEL:   %[[inc:.*]] = mulf %[[va]], %[[vb]] : f32
//       CHECKPARALLEL:   %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECKPARALLEL:   store %[[res]], %[[arg2]][%[[i0]], %[[i1]], %[[i2]]] : memref<?x?x?xf32>

#conv_4d_accesses = [
  affine_map<(m, n, k, l, m1, n1, k1, l1)[s0, s1, s2, s3] -> (m + m1 - s0 floordiv 2, n + n1 - s1 floordiv 2, k + k1 - s2 floordiv 2, l + l1 - s3 floordiv 2)>, // in
  affine_map<(m, n, k, l, m1, n1, k1, l1)[s0, s1, s2, s3] -> (m1, n1, k1, l1)>, // filter
  affine_map<(m, n, k, l, m1, n1, k1, l1)[s0, s1, s2, s3] -> (m, n, k, l)> // out
]

#conv_4d_trait = {
  args_in = 2,
  args_out = 1,
  doc = "C(m,n,k,l) += A(m,n,k,l) * B(m1,n1,k1,l1)",
  indexing_maps = #conv_4d_accesses,
  library_call  = "linalg_conv_4d",
  n_views = [2, 1],
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"],
  symbol_source = 1
}

func @conv4d(%in : memref<?x?x?x?xf32>, %filter : memref<?x?x?x?xf32>, %out :  memref<?x?x?x?xf32>) -> () {
  linalg.generic #conv_4d_trait %in, %filter, %out {
    ^bb0(%a: f32, %b: f32, %c: f32) :
      %d = mulf %a, %b : f32
      %e = addf %c, %d : f32
      linalg.yield %e : f32
  } : memref<?x?x?x?xf32>,
      memref<?x?x?x?xf32>,
      memref<?x?x?x?xf32>
  return
}

// CHECKLOOP-LABEL: @conv4d
//  CHECKLOOP-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//  CHECKLOOP-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//  CHECKLOOP-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//       CHECKLOOP: %[[c0:.*]] = constant 0 : index
//       CHECKLOOP: %[[c1:.*]] = constant 1 : index
//       CHECKLOOP: %[[c2:.*]] = constant 2 : index
//       CHECKLOOP: %[[c3:.*]] = constant 3 : index
//       CHECKLOOP: %[[dim0:.*]] = dim %[[arg1]], %[[c0]] : memref<?x?x?x?xf32>
//       CHECKLOOP: %[[dim1:.*]] = dim %[[arg1]], %[[c1]] : memref<?x?x?x?xf32>
//       CHECKLOOP: %[[dim2:.*]] = dim %[[arg1]], %[[c2]] : memref<?x?x?x?xf32>
//       CHECKLOOP: %[[dim3:.*]] = dim %[[arg1]], %[[c3]] : memref<?x?x?x?xf32>
//       CHECKLOOP: %[[dim4:.*]] = dim %[[arg2]], %[[c0]] : memref<?x?x?x?xf32>
//       CHECKLOOP: %[[dim5:.*]] = dim %[[arg2]], %[[c1]] : memref<?x?x?x?xf32>
//       CHECKLOOP: %[[dim6:.*]] = dim %[[arg2]], %[[c2]] : memref<?x?x?x?xf32>
//       CHECKLOOP: %[[dim7:.*]] = dim %[[arg2]], %[[c3]] : memref<?x?x?x?xf32>
//       CHECKLOOP: scf.for %[[i0:.*]] = %{{.*}} to %[[dim4]] step %{{.*}} {
//       CHECKLOOP:   scf.for %[[i1:.*]] = %{{.*}} to %[[dim5]] step %{{.*}} {
//       CHECKLOOP:     scf.for %[[i2:.*]] = %{{.*}} to %[[dim6]] step %{{.*}} {
//       CHECKLOOP:       scf.for %[[i3:.*]] = %{{.*}} to %[[dim7]] step %{{.*}} {
//       CHECKLOOP:         scf.for %[[i4:.*]] = %{{.*}} to %[[dim0]] step %{{.*}} {
//       CHECKLOOP:           scf.for %[[i5:.*]] = %{{.*}} to %[[dim1]] step %{{.*}} {
//       CHECKLOOP:             scf.for %[[i6:.*]] = %{{.*}} to %[[dim2]] step %{{.*}} {
//       CHECKLOOP:               scf.for %[[i7:.*]] = %{{.*}} to %[[dim3]] step %{{.*}} {
//       CHECKLOOP:                 %[[dim8:.*]] = dim %[[arg1]], %[[c0]] : memref<?x?x?x?xf32>
//       CHECKLOOP:                 %[[dim9:.*]] = dim %[[arg1]], %[[c1]] : memref<?x?x?x?xf32>
//       CHECKLOOP:                 %[[dim10:.*]] = dim %[[arg1]], %[[c2]] : memref<?x?x?x?xf32>
//       CHECKLOOP:                 %[[dim11:.*]] = dim %[[arg1]], %[[c3]] : memref<?x?x?x?xf32>
//       CHECKLOOP:                 %[[aff1:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim8]]]
//       CHECKLOOP:                 %[[aff2:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim9]]]
//       CHECKLOOP:                 %[[aff3:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim10]]]
//       CHECKLOOP:                 %[[aff4:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim11]]]
//       CHECKLOOP:                 %[[va:.*]] = load %[[arg0]][%[[aff1]], %[[aff2]], %[[aff3]], %[[aff4]]] : memref<?x?x?x?xf32>
//       CHECKLOOP:                 %[[vb:.*]] = load %[[arg1]][%[[i4]], %[[i5]], %[[i6]], %[[i7]]] : memref<?x?x?x?xf32>
//       CHECKLOOP:                 %[[vc:.*]] = load %[[arg2]][%[[i0]], %[[i1]], %[[i2]], %[[i3]]] : memref<?x?x?x?xf32>
//       CHECKLOOP:                 %[[inc:.*]] = mulf %[[va]], %[[vb]] : f32
//       CHECKLOOP:                 %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECKLOOP:                 store %[[res]], %[[arg2]][%[[i0]], %[[i1]], %[[i2]], %[[i3]]] : memref<?x?x?x?xf32>

// CHECKPARALLEL-LABEL: @conv4d
//  CHECKPARALLEL-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//  CHECKPARALLEL-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//  CHECKPARALLEL-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//       CHECKPARALLEL: %[[c0:.*]] = constant 0 : index
//       CHECKPARALLEL: %[[c1:.*]] = constant 1 : index
//       CHECKPARALLEL: %[[c2:.*]] = constant 2 : index
//       CHECKPARALLEL: %[[c3:.*]] = constant 3 : index
//       CHECKPARALLEL: %[[dim0:.*]] = dim %[[arg1]], %[[c0]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL: %[[dim1:.*]] = dim %[[arg1]], %[[c1]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL: %[[dim2:.*]] = dim %[[arg1]], %[[c2]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL: %[[dim3:.*]] = dim %[[arg1]], %[[c3]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL: %[[dim4:.*]] = dim %[[arg2]], %[[c0]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL: %[[dim5:.*]] = dim %[[arg2]], %[[c1]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL: %[[dim6:.*]] = dim %[[arg2]], %[[c2]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL: %[[dim7:.*]] = dim %[[arg2]], %[[c3]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL: scf.parallel (%[[i0:.*]], %[[i1:.*]], %[[i2:.*]], %[[i3:.*]], %[[i4:.*]], %[[i5:.*]], %[[i6:.*]], %[[i7:.*]]) = (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) to (%[[dim4]], %[[dim5]], %[[dim6]], %[[dim7]], %[[dim0]], %[[dim1]], %[[dim2]], %[[dim3]]) step ({{.*}}) {
//       CHECKPARALLEL:   %[[dim8:.*]] = dim %[[arg1]], %[[c0]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[dim9:.*]] = dim %[[arg1]], %[[c1]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[dim10:.*]] = dim %[[arg1]], %[[c2]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[dim11:.*]] = dim %[[arg1]], %[[c3]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[aff1:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim8]]]
//       CHECKPARALLEL:   %[[aff2:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim9]]]
//       CHECKPARALLEL:   %[[aff3:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim10]]]
//       CHECKPARALLEL:   %[[aff4:.*]] = affine.apply #[[$convMap]](%{{.*}}, %{{.*}})[%[[dim11]]]
//       CHECKPARALLEL:   %[[va:.*]] = load %[[arg0]][%[[aff1]], %[[aff2]], %[[aff3]], %[[aff4]]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[vb:.*]] = load %[[arg1]][%[[i4]], %[[i5]], %[[i6]], %[[i7]]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[vc:.*]] = load %[[arg2]][%[[i0]], %[[i1]], %[[i2]], %[[i3]]] : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[inc:.*]] = mulf %[[va]], %[[vb]] : f32
//       CHECKPARALLEL:   %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECKPARALLEL:   store %[[res]], %[[arg2]][%[[i0]], %[[i1]], %[[i2]], %[[i3]]] : memref<?x?x?x?xf32>
