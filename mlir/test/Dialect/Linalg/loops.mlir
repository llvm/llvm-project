// RUN: mlir-opt %s -convert-linalg-to-loops | FileCheck --check-prefix=CHECKLOOP %s
// RUN: mlir-opt %s -convert-linalg-to-parallel-loops | FileCheck --check-prefix=CHECKPARALLEL %s

// Test that we can lower all the way to LLVM without crashing, don't check results here.
// RUN: mlir-opt %s --convert-linalg-to-llvm -o=/dev/null 2>&1

// CHECKLOOP-DAG: #[[strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECKLOOP-DAG: #[[strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECKLOOP-DAG: #[[strided3D:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>
// CHECKLOOP-DAG: #[[strided4D:.*]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
// CHECKLOOP-DAG: #[[clampMinMap:.*]] = affine_map<(d0) -> (d0, 0)>

// CHECKLOOP-DAG: #[[Stride1Dilation1:.*]] = affine_map<(d0, d1) -> (d0  + d1)>
// CHECKLOOP-DAG: #[[Stride2Dilation1:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECKLOOP-DAG: #[[Stride2Dilation4:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1 * 4)>
// CHECKLOOP-DAG: #[[Stride3Dilation5:.*]] = affine_map<(d0, d1) -> (d0 * 3 + d1 * 5)>

// CHECKPARALLEL-DAG: #[[strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECKPARALLEL-DAG: #[[strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECKPARALLEL-DAG: #[[strided3D:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>
// CHECKPARALLEL-DAG: #[[strided4D:.*]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
// CHECKPARALLEL-DAG: #[[clampMinMap:.*]] = affine_map<(d0) -> (d0, 0)>

// CHECKPARALLEL-DAG: #[[Stride1Dilation1:.*]] = affine_map<(d0, d1) -> (d0  + d1)>
// CHECKPARALLEL-DAG: #[[Stride2Dilation1:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECKPARALLEL-DAG: #[[Stride2Dilation4:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1 * 4)>
// CHECKPARALLEL-DAG: #[[Stride3Dilation5:.*]] = affine_map<(d0, d1) -> (d0 * 3 + d1 * 5)>


func @matmul(%arg0: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %A = view %arg0[%c0][%M, %K] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %B = view %arg0[%c0][%K, %N] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %C = view %arg0[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
// CHECKLOOP-LABEL: func @matmul(%{{.*}}: memref<?xi8>,
//  CHECKLOOP-SAME: [[M:arg[0-9]+]]: index
//  CHECKLOOP-SAME: [[N:arg[0-9]+]]: index
//  CHECKLOOP-SAME: [[K:arg[0-9]+]]: index
//       CHECKLOOP: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECKLOOP: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECKLOOP: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECKLOOP: loop.for %{{.*}} = %{{.*}} to %[[M]] step %{{.*}} {
//       CHECKLOOP:   loop.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} {
//       CHECKLOOP:     loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKLOOP-DAG:       %[[a:.*]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECKLOOP-DAG:       %[[b:.*]] = load %[[B]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECKLOOP-DAG:       %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKLOOP-DAG:       %[[c:.*]] = load %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECKLOOP-DAG:       %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKLOOP:       store %[[res]], %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>

// CHECKPARALLEL-LABEL: func @matmul(%{{.*}}: memref<?xi8>,
//  CHECKPARALLEL-SAME: [[M:arg[0-9]+]]: index
//  CHECKPARALLEL-SAME: [[N:arg[0-9]+]]: index
//  CHECKPARALLEL-SAME: [[K:arg[0-9]+]]: index
//       CHECKPARALLEL: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECKPARALLEL: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECKPARALLEL: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECKPARALLEL: loop.parallel (%{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}) to (%[[M]], %[[N]]) step (%{{.*}}, %{{.*}} {
//       CHECKPARALLEL:   loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKPARALLEL-DAG:     %[[a:.*]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECKPARALLEL-DAG:     %[[b:.*]] = load %[[B]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECKPARALLEL-DAG:     %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKPARALLEL-DAG:     %[[c:.*]] = load %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECKPARALLEL-DAG:     %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKPARALLEL:     store %[[res]], %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>



func @matvec(%arg0: memref<?xi8>, %M: index, %N: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %2 = view %arg0[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %3 = view %arg0[%c0][%M] : memref<?xi8> to memref<?xf32, offset: ?, strides: [1]>
  %4 = view %arg0[%c0][%N] : memref<?xi8> to memref<?xf32, offset: ?, strides: [1]>
  linalg.matvec(%2, %3, %4) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECKLOOP-LABEL: func @matvec(%{{.*}}: memref<?xi8>,
//  CHECKLOOP-SAME: [[M:arg[0-9]+]]: index
//  CHECKLOOP-SAME: [[K:arg[0-9]+]]: index
//       CHECKLOOP: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECKLOOP: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECKLOOP: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECKLOOP: loop.for %{{.*}} = %{{.*}} to %[[M]] step %{{.*}} {
//       CHECKLOOP:   loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKLOOP-DAG:     %[[a:.*]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECKLOOP-DAG:     %[[b:.*]] = load %[[B]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECKLOOP-DAG:     %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKLOOP-DAG:     %[[c:.*]] = load %[[C]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECKLOOP-DAG:     %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKLOOP:     store %[[res]], %[[C]][%{{.*}}] : memref<?xf32, #[[strided1D]]>

// CHECKPARALLEL-LABEL: func @matvec(%{{.*}}: memref<?xi8>,
//  CHECKPARALLEL-SAME: [[M:arg[0-9]+]]: index
//  CHECKPARALLEL-SAME: [[K:arg[0-9]+]]: index
//       CHECKPARALLEL: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECKPARALLEL: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECKPARALLEL: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECKPARALLEL: loop.parallel (%{{.*}}) = (%{{.*}}) to (%[[M]]) step (%{{.*}}) {
//       CHECKPARALLEL:   loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKPARALLEL-DAG:     %[[a:.*]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECKPARALLEL-DAG:     %[[b:.*]] = load %[[B]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECKPARALLEL-DAG:     %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKPARALLEL-DAG:     %[[c:.*]] = load %[[C]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECKPARALLEL-DAG:     %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKPARALLEL:     store %[[res]], %[[C]][%{{.*}}] : memref<?xf32, #[[strided1D]]>


func @dot(%arg0: memref<?xi8>, %M: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %1 = view %arg0[%c0][%M] : memref<?xi8> to memref<?xf32, offset: ?, strides: [1]>
  %2 = view %arg0[%c0][%M] : memref<?xi8> to memref<?xf32, offset: ?, strides: [1]>
  %3 = view %arg0[][] : memref<?xi8> to memref<f32>
  linalg.dot(%1, %2, %3) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>, memref<f32>
  return
}
// CHECKLOOP-LABEL: func @dot(%{{.*}}: memref<?xi8>,
//  CHECKLOOP-SAME: [[K:arg[0-9]+]]: index
//       CHECKLOOP: %[[A:.*]] = std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECKLOOP: %[[B:.*]] = std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECKLOOP: %[[C:.*]] = std.view %{{.*}}[][] : memref<?xi8> to memref<f32>
//       CHECKLOOP: loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKLOOP-DAG:   %[[a:.*]] = load %[[A]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECKLOOP-DAG:   %[[b:.*]] = load %[[B]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECKLOOP-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKLOOP-DAG:   %[[c:.*]] = load %[[C]][] : memref<f32>
//   CHECKLOOP-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKLOOP:   store %[[res]], %[[C]][] : memref<f32>

// CHECKPARALLEL-LABEL: func @dot(%{{.*}}: memref<?xi8>,
//  CHECKPARALLEL-SAME: [[K:arg[0-9]+]]: index
//       CHECKPARALLEL: %[[A:.*]] = std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECKPARALLEL: %[[B:.*]] = std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECKPARALLEL: %[[C:.*]] = std.view %{{.*}}[][] : memref<?xi8> to memref<f32>
//       CHECKPARALLEL: loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKPARALLEL-DAG:   %[[a:.*]] = load %[[A]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECKPARALLEL-DAG:   %[[b:.*]] = load %[[B]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECKPARALLEL-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKPARALLEL-DAG:   %[[c:.*]] = load %[[C]][] : memref<f32>
//   CHECKPARALLEL-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKPARALLEL:   store %[[res]], %[[C]][] : memref<f32>


func @dot_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>, memref<f32>
  return
}
// CHECKLOOP-LABEL: func @dot_view(
//       CHECKLOOP:   %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<f32>) {
//       CHECKLOOP: %[[K:.*]] = dim %arg0, 0 : memref<?xf32, #[[strided1D]]>
//       CHECKLOOP: loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKLOOP-DAG:   %[[a:.*]] = load %arg0[%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECKLOOP-DAG:   %[[b:.*]] = load %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECKLOOP-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKLOOP-DAG:   %[[c:.*]] = load %{{.*}}[] : memref<f32>
//   CHECKLOOP-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKLOOP:   store %[[res]], %{{.*}}[] : memref<f32>

// CHECKPARALLEL-LABEL: func @dot_view(
//       CHECKPARALLEL:   %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<f32>) {
//       CHECKPARALLEL: %[[K:.*]] = dim %arg0, 0 : memref<?xf32, #[[strided1D]]>
//       CHECKPARALLEL: loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECKPARALLEL-DAG:   %[[a:.*]] = load %arg0[%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECKPARALLEL-DAG:   %[[b:.*]] = load %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECKPARALLEL-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKPARALLEL-DAG:   %[[c:.*]] = load %{{.*}}[] : memref<f32>
//   CHECKPARALLEL-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKPARALLEL:   store %[[res]], %{{.*}}[] : memref<f32>

func @fill_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<?xf32, offset: ?, strides: [1]>, f32
  return
}
// CHECKLOOP-LABEL: func @fill_view(
//       CHECKLOOP: %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: f32) {
//       CHECKLOOP:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:     store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>

// CHECKPARALLEL-LABEL: func @fill_view(
//       CHECKPARALLEL: %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: f32) {
//       CHECKPARALLEL:   loop.parallel (%{{.*}}) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
//       CHECKPARALLEL:     store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>

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
//       CHECKLOOP: %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: f32) {
//       CHECKLOOP:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:     loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:       loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:         store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>

// CHECKPARALLEL-LABEL: func @fill_view3(
//       CHECKPARALLEL: %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: f32) {
//       CHECKPARALLEL:   loop.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>

func @copy_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.copy(%arg0, %arg1) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECKLOOP-LABEL: func @copy_view(
//       CHECKLOOP: %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<?xf32, #[[strided1D]]>) {
//       CHECKLOOP:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:     %[[L:.*]] = load %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>
//       CHECKLOOP:     store %[[L]], %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>

// CHECKPARALLEL-LABEL: func @copy_view(
//       CHECKPARALLEL: %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<?xf32, #[[strided1D]]>) {
//       CHECKPARALLEL:   loop.parallel (%{{.*}}) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
//       CHECKPARALLEL:     %[[L:.*]] = load %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>
//       CHECKPARALLEL:     store %[[L]], %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>

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
//       CHECKLOOP: (%{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>) {
//       CHECKLOOP:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:     loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:       loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECKLOOP:         %[[L:.*]] = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKLOOP:         store %[[L]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>

// CHECKPARALLEL-LABEL: func @copy_view3
//       CHECKPARALLEL: (%{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>) {
//       CHECKPARALLEL:   loop.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     %[[L:.*]] = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKPARALLEL:     store %[[L]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>

func @conv_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {strides = [2]}: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECKLOOP-LABEL: func @conv_view3(
//       CHECKLOOP: %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>) {
//       CHECKLOOP:   %[[Z0:.*]] = dim %arg0, 0 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKLOOP:   %[[Q:.*]] = dim %arg0, 1 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKLOOP:   %[[K:.*]] = dim %arg0, 2 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKLOOP:   %[[B:.*]] = dim %arg1, 0 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKLOOP:   %[[X0:.*]] = dim %arg2, 1 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKLOOP:   loop.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       CHECKLOOP:     loop.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       CHECKLOOP:       loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECKLOOP:         loop.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECKLOOP:           loop.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECKLOOP:             %[[SUM:.*]] = affine.apply #[[Stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKLOOP:             %{{.*}} = load %{{.*}}[%{{.*}}, %[[SUM]], %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKLOOP:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKLOOP:             %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKLOOP:             %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:             store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>

// CHECKPARALLEL-LABEL: func @conv_view3(
//       CHECKPARALLEL: %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>) {
//       CHECKPARALLEL:   %[[Z0:.*]] = dim %arg0, 0 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKPARALLEL:   %[[Q:.*]] = dim %arg0, 1 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKPARALLEL:   %[[K:.*]] = dim %arg0, 2 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKPARALLEL:   %[[B:.*]] = dim %arg1, 0 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKPARALLEL:   %[[X0:.*]] = dim %arg2, 1 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKPARALLEL:   loop.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}) to (%[[B]], %[[X0]], %[[K]]) step (%{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     loop.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECKPARALLEL:       loop.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECKPARALLEL:         %[[SUM:.*]] = affine.apply #[[Stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:         %{{.*}} = load %{{.*}}[%{{.*}}, %[[SUM]], %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKPARALLEL:         %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKPARALLEL:         %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECKPARALLEL:         %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKPARALLEL:         %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECKPARALLEL:         store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>

func @conv_view4(%arg0: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg1: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg2: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [4, 5], strides = [2, 3]} : memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>
  return
}
// CHECKLOOP-LABEL: func @conv_view4(
//       CHECKLOOP: %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>) {
//       CHECKLOOP:   %[[Z0:.*]] = dim %arg0, 0 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKLOOP:   %[[Z1:.*]] = dim %arg0, 1 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKLOOP:   %[[Q:.*]] = dim %arg0, 2 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKLOOP:   %[[K:.*]] = dim %arg0, 3 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKLOOP:   %[[B:.*]] = dim %arg1, 0 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKLOOP:   %[[X0:.*]] = dim %arg2, 1 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKLOOP:   %[[X1:.*]] = dim %arg2, 2 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKLOOP:   loop.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       CHECKLOOP:     loop.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       CHECKLOOP:       loop.for %{{.*}} = %{{.*}} to %[[X1]] step %{{.*}} {
//       CHECKLOOP:         loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECKLOOP:           loop.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECKLOOP:             loop.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECKLOOP:               loop.for %{{.*}} = %{{.*}} to %[[Z1]] step %{{.*}} {
//       CHECKLOOP:                 %[[SUM0:.*]] = affine.apply #[[Stride2Dilation4]](%{{.*}}, %{{.*}})
//       CHECKLOOP:                 %[[SUM1:.*]] = affine.apply #[[Stride3Dilation5]](%{{.*}}, %{{.*}})
//       CHECKLOOP:                 %{{.*}} = load %{{.*}}[%{{.*}}, %[[SUM0]], %[[SUM1]], %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKLOOP:                 %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKLOOP:                 %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:                 %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKLOOP:                 %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:                 store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>

// CHECKPARALLEL-LABEL: func @conv_view4(
//       CHECKPARALLEL: %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>) {
//       CHECKPARALLEL:   %[[Z0:.*]] = dim %arg0, 0 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKPARALLEL:   %[[Z1:.*]] = dim %arg0, 1 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKPARALLEL:   %[[Q:.*]] = dim %arg0, 2 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKPARALLEL:   %[[K:.*]] = dim %arg0, 3 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKPARALLEL:   %[[B:.*]] = dim %arg1, 0 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKPARALLEL:   %[[X0:.*]] = dim %arg2, 1 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKPARALLEL:   %[[X1:.*]] = dim %arg2, 2 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKPARALLEL:   loop.parallel (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) to (%[[B]], %[[X0]], %[[X1]], %[[K]]) step (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     loop.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECKPARALLEL:       loop.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECKPARALLEL:         loop.for %{{.*}} = %{{.*}} to %[[Z1]] step %{{.*}} {
//       CHECKPARALLEL:           %[[SUM0:.*]] = affine.apply #[[Stride2Dilation4]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:           %[[SUM1:.*]] = affine.apply #[[Stride3Dilation5]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:           %{{.*}} = load %{{.*}}[%{{.*}}, %[[SUM0]], %[[SUM1]], %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKPARALLEL:           %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKPARALLEL:           %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECKPARALLEL:           %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECKPARALLEL:           %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECKPARALLEL:           store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>


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
//       CHECKLOOP:   %[[Z0:.*]] = dim %arg0, 0 : memref<?x?x?x?xf32>
//       CHECKLOOP:   %[[Z1:.*]] = dim %arg0, 1 : memref<?x?x?x?xf32>
//       CHECKLOOP:   %[[Q:.*]] =  dim %arg0, 2 : memref<?x?x?x?xf32>
//       CHECKLOOP:   %[[K:.*]] =  dim %arg0, 3 : memref<?x?x?x?xf32>
//       CHECKLOOP:   %[[B:.*]] =  dim %arg1, 0 : memref<?x?x?x?xf32>
//       CHECKLOOP:   %[[X0:.*]] = dim %arg2, 1 : memref<?x?x?x?xf32>
//       CHECKLOOP:   %[[X1:.*]] = dim %arg2, 2 : memref<?x?x?x?xf32>
//       CHECKLOOP:   loop.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       CHECKLOOP:     loop.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       CHECKLOOP:       loop.for %{{.*}} = %{{.*}} to %[[X1]] step %{{.*}} {
//       CHECKLOOP:         loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECKLOOP:           loop.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECKLOOP:             loop.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECKLOOP:               loop.for %{{.*}} = %{{.*}} to %[[Z1]] step %{{.*}} {
//       CHECKLOOP:                 %[[SUM0:.*]] = affine.apply #{{.*}}(%{{.*}}, %{{.*}})
//       CHECKLOOP:                 %[[SUM1:.*]] = affine.apply #{{.*}}(%{{.*}}, %{{.*}})
//       CHECKLOOP:                 %[[IDX:.*]] = affine.max #[[clampMinMap]](%[[SUM0]])
//       CHECKLOOP:                 %[[IDY:.*]] = affine.max #[[clampMinMap]](%[[SUM1]])
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
//       CHECKPARALLEL:   %[[Z0:.*]] = dim %arg0, 0 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[Z1:.*]] = dim %arg0, 1 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[Q:.*]] =  dim %arg0, 2 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[K:.*]] =  dim %arg0, 3 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[B:.*]] =  dim %arg1, 0 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[X0:.*]] = dim %arg2, 1 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[X1:.*]] = dim %arg2, 2 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   loop.parallel (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) to (%[[B]], %[[X0]], %[[X1]], %[[K]]) step (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     loop.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECKPARALLEL:       loop.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECKPARALLEL:         loop.for %{{.*}} = %{{.*}} to %[[Z1]] step %{{.*}} {
//       CHECKPARALLEL:           %[[SUM0:.*]] = affine.apply #{{.*}}(%{{.*}}, %{{.*}})
//       CHECKPARALLEL:           %[[SUM1:.*]] = affine.apply #{{.*}}(%{{.*}}, %{{.*}})
//       CHECKPARALLEL:           %[[IDX:.*]] = affine.max #[[clampMinMap]](%[[SUM0]])
//       CHECKPARALLEL:           %[[IDY:.*]] = affine.max #[[clampMinMap]](%[[SUM1]])
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
//       CHECKLOOP:   %[[WX:.*]] = dim %arg1, 0 : memref<?x?xi32>
//       CHECKLOOP:   %[[WY:.*]] = dim %arg1, 1 : memref<?x?xi32>
//       CHECKLOOP:   %[[OX:.*]] = dim %arg2, 0 : memref<?x?xf32>
//       CHECKLOOP:   %[[OY:.*]] = dim %arg2, 1 : memref<?x?xf32>
//       CHECKLOOP:   loop.for %{{.*}} = %{{.*}} to %[[OX]] step %{{.*}} {
//       CHECKLOOP:     loop.for %{{.*}} = %{{.*}} to %[[OY]] step %{{.*}} {
//       CHECKLOOP:       loop.for %{{.*}} = %{{.*}} to %[[WX]] step %{{.*}} {
//       CHECKLOOP:         loop.for %{{.*}} = %{{.*}} to %[[WY]] step %{{.*}} {
//       CHECKLOOP:           %[[IX:.*]] = affine.apply #[[Stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKLOOP:           %[[IY:.*]] = affine.apply #[[Stride1Dilation1]](%{{.*}}, %{{.*}})
//       CHECKLOOP:           %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
//       CHECKLOOP:           %{{.*}} = load %{{.*}}[%[[IX]], %[[IY]]] : memref<?x?xf32>
//       CHECKLOOP:           %[[RES:.*]] = select %{{.*}}, %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:           store %[[RES]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>

// CHECKPARALLEL-LABEL: func @pooling_max
//       CHECKPARALLEL:   %[[WX:.*]] = dim %arg1, 0 : memref<?x?xi32>
//       CHECKPARALLEL:   %[[WY:.*]] = dim %arg1, 1 : memref<?x?xi32>
//       CHECKPARALLEL:   %[[OX:.*]] = dim %arg2, 0 : memref<?x?xf32>
//       CHECKPARALLEL:   %[[OY:.*]] = dim %arg2, 1 : memref<?x?xf32>
//       CHECKPARALLEL:   loop.parallel (%{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}) to (%[[OX]], %[[OY]]) step (%{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     loop.for %{{.*}} = %{{.*}} to %[[WX]] step %{{.*}} {
//       CHECKPARALLEL:       loop.for %{{.*}} = %{{.*}} to %[[WY]] step %{{.*}} {
//       CHECKPARALLEL:         %[[IX:.*]] = affine.apply #[[Stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:         %[[IY:.*]] = affine.apply #[[Stride1Dilation1]](%{{.*}}, %{{.*}})
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
//       CHECKLOOP:   %[[WX:.*]] = dim %arg1, 0 : memref<?x?xi32>
//       CHECKLOOP:   %[[WY:.*]] = dim %arg1, 1 : memref<?x?xi32>
//       CHECKLOOP:   %[[OX:.*]] = dim %arg2, 0 : memref<?x?xf32>
//       CHECKLOOP:   %[[OY:.*]] = dim %arg2, 1 : memref<?x?xf32>
//       CHECKLOOP:   loop.for %{{.*}} = %{{.*}} to %[[OX]] step %{{.*}} {
//       CHECKLOOP:     loop.for %{{.*}} = %{{.*}} to %[[OY]] step %{{.*}} {
//       CHECKLOOP:       loop.for %{{.*}} = %{{.*}} to %[[WX]] step %{{.*}} {
//       CHECKLOOP:         loop.for %{{.*}} = %{{.*}} to %[[WY]] step %{{.*}} {
//       CHECKLOOP:           %[[IX:.*]] = affine.apply #[[Stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKLOOP:           %[[IY:.*]] = affine.apply #[[Stride1Dilation1]](%{{.*}}, %{{.*}})
//       CHECKLOOP:           %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
//       CHECKLOOP:           %{{.*}} = load %{{.*}}[%[[IX]], %[[IY]]] : memref<?x?xf32>
//       CHECKLOOP:           %[[RES:.*]] = select %{{.*}}, %{{.*}}, %{{.*}} : f32
//       CHECKLOOP:           store %[[RES]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>

// CHECKPARALLEL-LABEL: func @pooling_min
//       CHECKPARALLEL:   %[[WX:.*]] = dim %arg1, 0 : memref<?x?xi32>
//       CHECKPARALLEL:   %[[WY:.*]] = dim %arg1, 1 : memref<?x?xi32>
//       CHECKPARALLEL:   %[[OX:.*]] = dim %arg2, 0 : memref<?x?xf32>
//       CHECKPARALLEL:   %[[OY:.*]] = dim %arg2, 1 : memref<?x?xf32>
//       CHECKPARALLEL:   loop.parallel (%{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}) to (%[[OX]], %[[OY]]) step (%{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     loop.for %{{.*}} = %{{.*}} to %[[WX]] step %{{.*}} {
//       CHECKPARALLEL:       loop.for %{{.*}} = %{{.*}} to %[[WY]] step %{{.*}} {
//       CHECKPARALLEL:         %[[IX:.*]] = affine.apply #[[Stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:         %[[IY:.*]] = affine.apply #[[Stride1Dilation1]](%{{.*}}, %{{.*}})
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
//       CHECKLOOP:   %[[WX:.*]] = dim %arg1, 0 : memref<?x?xi32>
//       CHECKLOOP:   %[[WY:.*]] = dim %arg1, 1 : memref<?x?xi32>
//       CHECKLOOP:   %[[OX:.*]] = dim %arg2, 0 : memref<?x?xf32>
//       CHECKLOOP:   %[[OY:.*]] = dim %arg2, 1 : memref<?x?xf32>
//       CHECKLOOP:   loop.for %{{.*}} = %{{.*}} to %[[OX]] step %{{.*}} {
//       CHECKLOOP:     loop.for %{{.*}} = %{{.*}} to %[[OY]] step %{{.*}} {
//       CHECKLOOP:       loop.for %{{.*}} = %{{.*}} to %[[WX]] step %{{.*}} {
//       CHECKLOOP:         loop.for %{{.*}} = %{{.*}} to %[[WY]] step %{{.*}} {
//       CHECKLOOP:           %[[IX:.*]] = affine.apply #[[Stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKLOOP:           %[[IY:.*]] = affine.apply #[[Stride1Dilation1]](%{{.*}}, %{{.*}})
//       CHECKLOOP:           %[[RHS:.*]] = load %{{.*}}[%[[IX]], %[[IY]]] : memref<?x?xf32>
//       CHECKLOOP:           %[[LHS:.*]] = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
//       CHECKLOOP:           %[[RES:.*]] = addf %[[LHS]], %[[RHS]] : f32
//       CHECKLOOP:           store %[[RES]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>

// CHECKPARALLEL-LABEL: func @pooling_sum
//       CHECKPARALLEL:   %[[WX:.*]] = dim %arg1, 0 : memref<?x?xi32>
//       CHECKPARALLEL:   %[[WY:.*]] = dim %arg1, 1 : memref<?x?xi32>
//       CHECKPARALLEL:   %[[OX:.*]] = dim %arg2, 0 : memref<?x?xf32>
//       CHECKPARALLEL:   %[[OY:.*]] = dim %arg2, 1 : memref<?x?xf32>
//       CHECKPARALLEL:   loop.parallel (%{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}) to (%[[OX]], %[[OY]]) step (%{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     loop.for %{{.*}} = %{{.*}} to %[[WX]] step %{{.*}} {
//       CHECKPARALLEL:       loop.for %{{.*}} = %{{.*}} to %[[WY]] step %{{.*}} {
//       CHECKPARALLEL:         %[[IX:.*]] = affine.apply #[[Stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECKPARALLEL:         %[[IY:.*]] = affine.apply #[[Stride1Dilation1]](%{{.*}}, %{{.*}})
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
//       CHECKLOOP: loop.for %[[i:.*]] = {{.*}}
//       CHECKLOOP:   loop.for %[[j:.*]] = {{.*}}
//       CHECKLOOP:     loop.for %[[k:.*]] = {{.*}}
//       CHECKLOOP:       %[[a:.*]] = load %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32, #[[strided2D]]>
//       CHECKLOOP:       %[[b:.*]] = load %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKLOOP:       %[[c:.*]] = load %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKLOOP:       %[[d:.*]] = mulf %[[a]], %[[b]] : f32
//       CHECKLOOP:       %[[e:.*]] = addf %[[c]], %[[d]] : f32
//       CHECKLOOP:       store %[[d]], %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKLOOP:       store %[[e]], %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[strided3D]]>

// CHECKPARALLEL-LABEL: @generic_region
//       CHECKPARALLEL: loop.parallel (%[[i:[a-zA-Z0-9_]*]], %[[j:[a-zA-Z0-9_]*]], %[[k:[a-zA-Z0-9_]*]])
//       CHECKPARALLEL:   %[[a:.*]] = load %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32, #[[strided2D]]>
//       CHECKPARALLEL:   %[[b:.*]] = load %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKPARALLEL:   %[[c:.*]] = load %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKPARALLEL:   %[[d:.*]] = mulf %[[a]], %[[b]] : f32
//       CHECKPARALLEL:   %[[e:.*]] = addf %[[c]], %[[d]] : f32
//       CHECKPARALLEL:   store %[[d]], %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECKPARALLEL:   store %[[e]], %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[strided3D]]>

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
//       CHECKLOOP: loop.for %[[i:.*]] = {{.*}}
//       CHECKLOOP:   loop.for %[[j:.*]] = {{.*}}
//       CHECKLOOP:     loop.for %[[k:.*]] = {{.*}}
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
//       CHECKPARALLEL: loop.parallel (%[[i:[a-zA-Z0-9_]*]], %[[j:[a-zA-Z0-9_]*]], %[[k:[a-zA-Z0-9_]*]])
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
//       CHECKLOOP: loop.for %[[i:.*]] = {{.*}}
//       CHECKLOOP:   loop.for %[[j:.*]] = {{.*}}
//       CHECKLOOP:     %[[a:.*]] = load %[[ARG0]][]
//       CHECKLOOP:     store %[[a]], %[[ARG1]][%[[i]], %[[j]]]

// CHECKPARALLEL-LABEL: @generic_op_zero_rank
//  CHECKPARALLEL-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECKPARALLEL-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<3x4xf32>
//       CHECKPARALLEL: loop.parallel (%[[i:[a-zA-Z0-9_]*]], %[[j:[a-zA-Z0-9_]*]])
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
//       CHECKLOOP: loop.for %[[i:.*]] = {{.*}}
//       CHECKLOOP:   loop.for %[[j:.*]] = {{.*}}
//       CHECKLOOP:     %[[a:.*]] = load %[[ARG0]][
//       CHECKLOOP:     %[[ij:.*]] = addi %[[i]], %[[j]] : index
//       CHECKLOOP:     %[[ij_int:.*]] = index_cast %[[ij]] : index to i32
//       CHECKLOOP:     %[[result:.*]] = addi %[[a]], %[[ij_int]] : i32
//       CHECKLOOP:     store %[[result]], %[[ARG1]][%[[i]], %[[j]]]

// CHECKPARALLEL-LABEL: @indexed_generic_op_zero_rank
//  CHECKPARALLEL-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<i32>
//  CHECKPARALLEL-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<3x4xi32>
//       CHECKPARALLEL: loop.parallel (%[[i:[a-zA-Z0-9_]*]], %[[j:[a-zA-Z0-9_]*]])
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
//       CHECKLOOP: loop.for %[[i:.*]] = {{.*}}
//       CHECKLOOP:   %[[a:.*]] = load %[[ARG0]][%[[i]]]
//       CHECKLOOP:   %[[b:.*]] = load %[[ARG1]][]
//       CHECKLOOP:   %[[c:.*]] = addf %[[a]], %[[b]] : f32
//       CHECKLOOP:   store %[[c]], %[[ARG1]][]

// CHECKPARALLEL-LABEL: @generic_op_1D_reduce
//  CHECKPARALLEL-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?xf32>
//  CHECKPARALLEL-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
//       CHECKPARALLEL: loop.for %[[i:.*]] = {{.*}}
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
//       CHECKLOOP: loop.for %[[i:.*]] = {{.*}}
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
//       CHECKPARALLEL: loop.for %[[i:.*]] = {{.*}}
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
//       CHECKLOOP: loop.for %[[i:.*]] = {{.*}}
//       CHECKLOOP:   store %[[CONST]], %[[ARG0]]

// CHECKPARALLEL-LABEL: @generic_const_init
//  CHECKPARALLEL-SAME: %[[ARG0:.*]]: memref<?xf32>
//       CHECKPARALLEL: %[[CONST:.*]] = constant 1.000000e+00 : f32
//       CHECKPARALLEL: loop.parallel (%[[i:.*]])
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
//  CHECKLOOP-SAME: %[[ARG0]]: memref<f32>
//  CHECKLOOP-SAME: %[[ARG1]]: memref<f32>
//  CHECKLOOP-SAME: %[[ARG2]]: memref<f32>
//   CHECKLOOP-NOT: loop.for
//   CHECKLOOP-DAG: load %[[ARG0]][]
//   CHECKLOOP-DAG: load %[[ARG1]][]
//   CHECKLOOP-DAG: load %[[ARG2]][]
//       CHECKLOOP: addf
//       CHECKLOOP: store %{{.*}}, %[[ARG2]][]

// CHECKPARALLEL-LABEL: @scalar_code
//  CHECKPARALLEL-SAME: %[[ARG0]]: memref<f32>
//  CHECKPARALLEL-SAME: %[[ARG1]]: memref<f32>
//  CHECKPARALLEL-SAME: %[[ARG2]]: memref<f32>
//   CHECKPARALLEL-NOT: loop.for
//   CHECKPARALLEL-DAG: load %[[ARG0]][]
//   CHECKPARALLEL-DAG: load %[[ARG1]][]
//   CHECKPARALLEL-DAG: load %[[ARG2]][]
//       CHECKPARALLEL: addf
//       CHECKPARALLEL: store %{{.*}}, %[[ARG2]][]
