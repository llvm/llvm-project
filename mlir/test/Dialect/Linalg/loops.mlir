// RUN: mlir-opt %s -convert-linalg-to-loops | FileCheck %s
// RUN: mlir-opt %s -convert-linalg-to-parallel-loops | FileCheck --check-prefix=CHECKPARALLEL %s

// Test that we can lower all the way to LLVM without crashing, don't check results here.
// RUN: mlir-opt %s -convert-linalg-to-loops -convert-linalg-to-llvm -o=/dev/null 2>&1

// CHECK-DAG: #[[$strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[$strided3D:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>
// CHECK-DAG: #[[$strided4D:.*]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
// CHECK-DAG: #[[$clampMinMap:.*]] = affine_map<(d0) -> (d0, 0)>

// CHECK-DAG: #[[$stride1Dilation1:.*]] = affine_map<(d0, d1) -> (d0  + d1)>
// CHECK-DAG: #[[$stride2Dilation1:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK-DAG: #[[$stride2Dilation4:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1 * 4)>
// CHECK-DAG: #[[$stride3Dilation5:.*]] = affine_map<(d0, d1) -> (d0 * 3 + d1 * 5)>

// CHECKPARALLEL-DAG: #[[$strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECKPARALLEL-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECKPARALLEL-DAG: #[[$strided3D:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>
// CHECKPARALLEL-DAG: #[[$strided4D:.*]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
// CHECKPARALLEL-DAG: #[[$clampMinMap:.*]] = affine_map<(d0) -> (d0, 0)>

// CHECKPARALLEL-DAG: #[[$stride1Dilation1:.*]] = affine_map<(d0, d1) -> (d0  + d1)>
// CHECKPARALLEL-DAG: #[[$stride2Dilation1:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECKPARALLEL-DAG: #[[$stride2Dilation4:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1 * 4)>
// CHECKPARALLEL-DAG: #[[$stride3Dilation5:.*]] = affine_map<(d0, d1) -> (d0 * 3 + d1 * 5)>

func @matmul(%arg0: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %A = memref.view %arg0[%c0][%M, %K] : memref<?xi8> to memref<?x?xf32>
  %B = memref.view %arg0[%c0][%K, %N] : memref<?xi8> to memref<?x?xf32>
  %C = memref.view %arg0[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32>
  linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
               outs(%C: memref<?x?xf32>)
  return
}
// CHECK-LABEL: func @matmul(%{{.*}}: memref<?xi8>,
//  CHECK-SAME: [[M:arg[0-9]+]]: index
//  CHECK-SAME: [[N:arg[0-9]+]]: index
//  CHECK-SAME: [[K:arg[0-9]+]]: index
//       CHECK: %[[A:.*]] = memref.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECK: %[[B:.*]] = memref.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECK: %[[C:.*]] = memref.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECK: scf.for {{.*}} to %[[M]]
//       CHECK:   scf.for {{.*}} to %[[N]]
//       CHECK:     scf.for {{.*}} to %[[K]]
//   CHECK-DAG:       %[[a:.*]] = memref.load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECK-DAG:       %[[b:.*]] = memref.load %[[B]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECK-DAG:       %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:       %[[c:.*]] = memref.load %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECK-DAG:       %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:       store %[[res]], %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32>

// CHECKPARALLEL-LABEL: func @matmul(%{{.*}}: memref<?xi8>,
//  CHECKPARALLEL-SAME: [[M:arg[0-9]+]]: index
//  CHECKPARALLEL-SAME: [[N:arg[0-9]+]]: index
//  CHECKPARALLEL-SAME: [[K:arg[0-9]+]]: index
//       CHECKPARALLEL: %[[A:.*]] = memref.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECKPARALLEL: %[[B:.*]] = memref.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECKPARALLEL: %[[C:.*]] = memref.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECKPARALLEL: scf.parallel {{.*}} to (%[[M]], %[[N]]) step (%{{.*}}, %{{.*}} {
//       CHECKPARALLEL:   scf.for {{.*}} to %[[K]]
//   CHECKPARALLEL-DAG:     %[[a:.*]] = memref.load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECKPARALLEL-DAG:     %[[b:.*]] = memref.load %[[B]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECKPARALLEL-DAG:     %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKPARALLEL-DAG:     %[[c:.*]] = memref.load %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECKPARALLEL-DAG:     %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKPARALLEL:     store %[[res]], %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32>



func @matvec(%arg0: memref<?xi8>, %M: index, %N: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %2 = memref.view %arg0[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32>
  %3 = memref.view %arg0[%c0][%M] : memref<?xi8> to memref<?xf32>
  %4 = memref.view %arg0[%c0][%N] : memref<?xi8> to memref<?xf32>
  linalg.matvec ins(%2, %3: memref<?x?xf32>, memref<?xf32>)
               outs(%4 : memref<?xf32>)
  return
}
// CHECK-LABEL: func @matvec(%{{.*}}: memref<?xi8>,
//  CHECK-SAME: [[M:arg[0-9]+]]: index
//  CHECK-SAME: [[K:arg[0-9]+]]: index
//       CHECK: %[[A:.*]] = memref.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECK: %[[B:.*]] = memref.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECK: %[[C:.*]] = memref.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECK: scf.for {{.*}} to %[[M]]
//       CHECK:   scf.for {{.*}} to %[[K]]
//   CHECK-DAG:     %[[a:.*]] = memref.load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECK-DAG:     %[[b:.*]] = memref.load %[[B]][%{{.*}}] : memref<?xf32>
//   CHECK-DAG:     %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:     %[[c:.*]] = memref.load %[[C]][%{{.*}}] : memref<?xf32>
//   CHECK-DAG:     %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:     store %[[res]], %[[C]][%{{.*}}] : memref<?xf32>

// CHECKPARALLEL-LABEL: func @matvec(%{{.*}}: memref<?xi8>,
//  CHECKPARALLEL-SAME: [[M:arg[0-9]+]]: index
//  CHECKPARALLEL-SAME: [[K:arg[0-9]+]]: index
//       CHECKPARALLEL: %[[A:.*]] = memref.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECKPARALLEL: %[[B:.*]] = memref.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECKPARALLEL: %[[C:.*]] = memref.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECKPARALLEL: scf.parallel (%{{.*}}) = (%{{.*}}) to (%[[M]]) step (%{{.*}}) {
//       CHECKPARALLEL:   scf.for {{.*}} to %[[K]]
//   CHECKPARALLEL-DAG:     %[[a:.*]] = memref.load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32>
//   CHECKPARALLEL-DAG:     %[[b:.*]] = memref.load %[[B]][%{{.*}}] : memref<?xf32>
//   CHECKPARALLEL-DAG:     %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKPARALLEL-DAG:     %[[c:.*]] = memref.load %[[C]][%{{.*}}] : memref<?xf32>
//   CHECKPARALLEL-DAG:     %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKPARALLEL:     store %[[res]], %[[C]][%{{.*}}] : memref<?xf32>


func @dot(%arg0: memref<?xi8>, %M: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %1 = memref.view %arg0[%c0][%M] : memref<?xi8> to memref<?xf32>
  %2 = memref.view %arg0[%c0][%M] : memref<?xi8> to memref<?xf32>
  %3 = memref.view %arg0[%c0][] : memref<?xi8> to memref<f32>
  linalg.dot ins(%1, %2 : memref<?xf32>, memref<?xf32>)
            outs(%3 : memref<f32>)
  return
}
// CHECK-LABEL: func @dot(%{{.*}}: memref<?xi8>,
//  CHECK-SAME: [[K:arg[0-9]+]]: index
//       CHECK: %[[A:.*]] = memref.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECK: %[[B:.*]] = memref.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECK: %[[C:.*]] = memref.view %{{.*}}[{{.*}}][] : memref<?xi8> to memref<f32>
//       CHECK: scf.for {{.*}} to %[[K]]
//   CHECK-DAG:   %[[a:.*]] = memref.load %[[A]][%{{.*}}] : memref<?xf32>
//   CHECK-DAG:   %[[b:.*]] = memref.load %[[B]][%{{.*}}] : memref<?xf32>
//   CHECK-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:   %[[c:.*]] = memref.load %[[C]][] : memref<f32>
//   CHECK-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:   store %[[res]], %[[C]][] : memref<f32>

// CHECKPARALLEL-LABEL: func @dot(%{{.*}}: memref<?xi8>,
//  CHECKPARALLEL-SAME: [[K:arg[0-9]+]]: index
//       CHECKPARALLEL: %[[A:.*]] = memref.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECKPARALLEL: %[[B:.*]] = memref.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32>
//       CHECKPARALLEL: %[[C:.*]] = memref.view %{{.*}}[{{.*}}][] : memref<?xi8> to memref<f32>
//       CHECKPARALLEL: scf.for {{.*}} to %[[K]]
//   CHECKPARALLEL-DAG:   %[[a:.*]] = memref.load %[[A]][%{{.*}}] : memref<?xf32>
//   CHECKPARALLEL-DAG:   %[[b:.*]] = memref.load %[[B]][%{{.*}}] : memref<?xf32>
//   CHECKPARALLEL-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKPARALLEL-DAG:   %[[c:.*]] = memref.load %[[C]][] : memref<f32>
//   CHECKPARALLEL-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKPARALLEL:   store %[[res]], %[[C]][] : memref<f32>


func @dot_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<f32>) {
  linalg.dot ins(%arg0, %arg1 : memref<?xf32, offset: ?, strides: [1]>,
                                memref<?xf32, offset: ?, strides: [1]>)
            outs(%arg2:  memref<f32>)
  return
}
// CHECK-LABEL: func @dot_view(
//       CHECK:   %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: memref<f32>) {
//       CHECK: %[[K:.*]] = memref.dim %arg0, %c0 : memref<?xf32, #[[$strided1D]]>
//       CHECK: scf.for {{.*}} to %[[K]]
//   CHECK-DAG:   %[[a:.*]] = memref.load %arg0[%{{.*}}] : memref<?xf32, #[[$strided1D]]>
//   CHECK-DAG:   %[[b:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>
//   CHECK-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:   %[[c:.*]] = memref.load %{{.*}}[] : memref<f32>
//   CHECK-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:   store %[[res]], %{{.*}}[] : memref<f32>

// CHECKPARALLEL-LABEL: func @dot_view(
//       CHECKPARALLEL:   %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: memref<f32>) {
//       CHECKPARALLEL: %[[K:.*]] = memref.dim %arg0, %c0 : memref<?xf32, #[[$strided1D]]>
//       CHECKPARALLEL: scf.for {{.*}} to %[[K]]
//   CHECKPARALLEL-DAG:   %[[a:.*]] = memref.load %arg0[%{{.*}}] : memref<?xf32, #[[$strided1D]]>
//   CHECKPARALLEL-DAG:   %[[b:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>
//   CHECKPARALLEL-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECKPARALLEL-DAG:   %[[c:.*]] = memref.load %{{.*}}[] : memref<f32>
//   CHECKPARALLEL-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECKPARALLEL:   store %[[res]], %{{.*}}[] : memref<f32>

func @fill_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: f32) {
  linalg.fill(%arg1, %arg0) : f32, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @fill_view(
//       CHECK: %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: f32) {
//       CHECK:   scf.for {{.*}} to %{{.*}}
//       CHECK:     store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>

// CHECKPARALLEL-LABEL: func @fill_view(
//       CHECKPARALLEL: %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: f32) {
//       CHECKPARALLEL:   scf.parallel (%{{.*}}) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
//       CHECKPARALLEL:     store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>

func @fill_view0(%arg0: memref<f32>, %arg1: f32) {
  linalg.fill(%arg1, %arg0) : f32, memref<f32>
  return
}
// CHECK-LABEL: func @fill_view0(%{{.*}}: memref<f32>, %{{.*}}: f32) {
//       CHECK:   store %{{.*}}, %{{.*}}[] : memref<f32>

// CHECKPARALLEL-LABEL: func @fill_view0(%{{.*}}: memref<f32>, %{{.*}}: f32) {
//       CHECKPARALLEL:   store %{{.*}}, %{{.*}}[] : memref<f32>

func @fill_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: f32) {
  linalg.fill(%arg1, %arg0) : f32, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @fill_view3(
//       CHECK: %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: f32) {
//       CHECK:   scf.for {{.*}} to %{{.*}}
//       CHECK:     scf.for {{.*}} to %{{.*}}
//       CHECK:       scf.for {{.*}} to %{{.*}}
//       CHECK:         store %{{.*}}, {{.*}} : memref<?x?x?xf32, #[[$strided3D]]>

// CHECKPARALLEL-LABEL: func @fill_view3(
//       CHECKPARALLEL: %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: f32) {
//       CHECKPARALLEL:   scf.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     store %{{.*}}, {{.*}} : memref<?x?x?xf32, #[[$strided3D]]>

func @copy_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.copy(%arg0, %arg1) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @copy_view(
//       CHECK: %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: memref<?xf32, #[[$strided1D]]>) {
//       CHECK:   scf.for {{.*}} to %{{.*}}
//       CHECK:     %[[L:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>
//       CHECK:     store %[[L]], %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>

// CHECKPARALLEL-LABEL: func @copy_view(
//       CHECKPARALLEL: %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: memref<?xf32, #[[$strided1D]]>) {
//       CHECKPARALLEL:   scf.parallel (%{{.*}}) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
//       CHECKPARALLEL:     %[[L:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>
//       CHECKPARALLEL:     store %[[L]], %{{.*}}[%{{.*}}] : memref<?xf32, #[[$strided1D]]>

func @copy_view0(%arg0: memref<f32>, %arg1: memref<f32>) {
  linalg.copy(%arg0, %arg1) : memref<f32>, memref<f32>
  return
}
// CHECK-LABEL: func @copy_view0(%{{.*}}: memref<f32>, %{{.*}}: memref<f32>) {
//       CHECK:   memref.load %{{.*}}[] : memref<f32>
//       CHECK:   store %{{.*}}, %{{.*}}[] : memref<f32>

// CHECKPARALLEL-LABEL: func @copy_view0(%{{.*}}: memref<f32>, %{{.*}}: memref<f32>) {
//       CHECKPARALLEL:   memref.load %{{.*}}[] : memref<f32>
//       CHECKPARALLEL:   store %{{.*}}, %{{.*}}[] : memref<f32>

func @copy_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.copy(%arg0, %arg1) {inputPermutation = affine_map<(i, j, k) -> (i, k, j)>,
                             outputPermutation = affine_map<(i, j, k) -> (k, j, i)>} :
    memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @copy_view3
//       CHECK: (%{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>) {
//       CHECK:   scf.for {{.*}} to %{{.*}}
//       CHECK:     scf.for {{.*}} to %{{.*}}
//       CHECK:       scf.for {{.*}} to %{{.*}}
//       CHECK:         %[[L:.*]] = memref.load {{.*}} : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:         store %[[L]], {{.*}} : memref<?x?x?xf32, #[[$strided3D]]>

// CHECKPARALLEL-LABEL: func @copy_view3
//       CHECKPARALLEL: (%{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>) {
//       CHECKPARALLEL:   scf.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     %[[L:.*]] = memref.load {{.*}} : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:     store %[[L]], {{.*}} : memref<?x?x?xf32, #[[$strided3D]]>

func @conv_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {strides = [2]}: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @conv_view3(
//       CHECK: %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>) {
//       CHECK:   %[[Z0:.*]] = memref.dim %arg0, %c0 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:   %[[Q:.*]] = memref.dim %arg0, %c1 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:   %[[K:.*]] = memref.dim %arg0, %c2 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:   %[[B:.*]] = memref.dim %arg1, %c0 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:   %[[X0:.*]] = memref.dim %arg2, %c1 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:   scf.for {{.*}} to %[[B]]
//       CHECK:     scf.for {{.*}} to %[[X0]]
//       CHECK:       scf.for {{.*}} to %[[K]]
//       CHECK:         scf.for {{.*}} to %[[Q]]
//       CHECK:           scf.for {{.*}} to %[[Z0]]
//       CHECK:             %[[SUM:.*]] = affine.apply #[[$stride2Dilation1]]
//       CHECK:             memref.load %{{.*}}[%{{.*}}, %[[SUM]], %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:             memref.load {{.*}} : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:             mulf
//       CHECK:             memref.load {{.*}} : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:             addf
//       CHECK:             store %{{.*}}, {{.*}} : memref<?x?x?xf32, #[[$strided3D]]>

// CHECKPARALLEL-LABEL: func @conv_view3(
//       CHECKPARALLEL: %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>) {
//       CHECKPARALLEL:   %[[Z0:.*]] = memref.dim %arg0, %c0 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:   %[[Q:.*]] = memref.dim %arg0, %c1 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:   %[[K:.*]] = memref.dim %arg0, %c2 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:   %[[B:.*]] = memref.dim %arg1, %c0 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:   %[[X0:.*]] = memref.dim %arg2, %c1 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:   scf.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}) to (%[[B]], %[[X0]], %[[K]]) step (%{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     scf.for {{.*}} to %[[Q]]
//       CHECKPARALLEL:       scf.for {{.*}} to %[[Z0]]
//       CHECKPARALLEL:         %[[SUM:.*]] = affine.apply #[[$stride2Dilation1]]
//       CHECKPARALLEL:         memref.load %{{.*}}[%{{.*}}, %[[SUM]], %{{.*}}] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:         memref.load {{.*}} : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:         mulf
//       CHECKPARALLEL:         memref.load {{.*}} : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:         addf
//       CHECKPARALLEL:         store %{{.*}}, {{.*}} : memref<?x?x?xf32, #[[$strided3D]]>

func @conv_view4(%arg0: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg1: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg2: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [4, 5], strides = [2, 3]} : memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>
  return
}
// CHECK-LABEL: func @conv_view4(
//       CHECK: %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>) {
//       CHECK:   %[[Z0:.*]] = memref.dim %arg0, %c0 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECK:   %[[Z1:.*]] = memref.dim %arg0, %c1 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECK:   %[[Q:.*]] = memref.dim %arg0, %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECK:   %[[K:.*]] = memref.dim %arg0, %c3 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECK:   %[[B:.*]] = memref.dim %arg1, %c0 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECK:   %[[X0:.*]] = memref.dim %arg2, %c1 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECK:   %[[X1:.*]] = memref.dim %arg2, %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECK:   scf.for {{.*}} to %[[B]]
//       CHECK:     scf.for {{.*}} to %[[X0]]
//       CHECK:       scf.for {{.*}} to %[[X1]]
//       CHECK:         scf.for {{.*}} to %[[K]]
//       CHECK:           scf.for {{.*}} to %[[Q]]
//       CHECK:             scf.for {{.*}} to %[[Z0]]
//       CHECK:               scf.for {{.*}} to %[[Z1]]
//       CHECK:                 %[[SUM0:.*]] = affine.apply #[[$stride2Dilation4]]
//       CHECK:                 %[[SUM1:.*]] = affine.apply #[[$stride3Dilation5]]
//       CHECK:                 memref.load %{{.*}}[%{{.*}}, %[[SUM0]], %[[SUM1]], %{{.*}}] : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECK:                 memref.load {{.*}} : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECK:                 mulf
//       CHECK:                 memref.load {{.*}} : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECK:                 addf
//       CHECK:                 store %{{.*}}, {{.*}} : memref<?x?x?x?xf32, #[[$strided4D]]>

// CHECKPARALLEL-LABEL: func @conv_view4(
//       CHECKPARALLEL: %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[$strided4D]]>) {
//       CHECKPARALLEL:   %[[Z0:.*]] = memref.dim %arg0, %c0 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:   %[[Z1:.*]] = memref.dim %arg0, %c1 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:   %[[Q:.*]] = memref.dim %arg0, %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:   %[[K:.*]] = memref.dim %arg0, %c3 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:   %[[B:.*]] = memref.dim %arg1, %c0 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:   %[[X0:.*]] = memref.dim %arg2, %c1 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:   %[[X1:.*]] = memref.dim %arg2, %c2 : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:   scf.parallel (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) to (%[[B]], %[[X0]], %[[X1]], %[[K]]) step (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     scf.for {{.*}} to %[[Q]]
//       CHECKPARALLEL:       scf.for {{.*}} to %[[Z0]]
//       CHECKPARALLEL:         scf.for {{.*}} to %[[Z1]]
//       CHECKPARALLEL:           %[[SUM0:.*]] = affine.apply #[[$stride2Dilation4]]
//       CHECKPARALLEL:           %[[SUM1:.*]] = affine.apply #[[$stride3Dilation5]]
//       CHECKPARALLEL:           memref.load %{{.*}}[%{{.*}}, %[[SUM0]], %[[SUM1]], %{{.*}}] : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:           memref.load {{.*}} : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:           mulf
//       CHECKPARALLEL:           memref.load {{.*}} : memref<?x?x?x?xf32, #[[$strided4D]]>
//       CHECKPARALLEL:           addf
//       CHECKPARALLEL:           store %{{.*}}, {{.*}} : memref<?x?x?x?xf32, #[[$strided4D]]>

func @conv_padding(%arg0: memref<?x?x?x?xf32>,
                   %arg1: memref<?x?x?x?xf32>,
                   %arg2: memref<?x?x?x?xf32>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [1, 1],
                                    padding = dense<[[0, 1], [1, 1]]> : tensor<2x2xi64>,
                                    strides = [1, 1]} :
    memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
  return
}
// CHECK-LABEL: func @conv_padding
//       CHECK: %{{.*}}: memref<?x?x?x?xf32>, %{{.*}}: memref<?x?x?x?xf32>, %{{.*}}: memref<?x?x?x?xf32>) {
//       CHECK:   %[[ZERO:.*]] = constant 0.000000e+00 : f32
//       CHECK:   %[[Z0:.*]] = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
//       CHECK:   %[[Z1:.*]] = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
//       CHECK:   %[[Q:.*]] =  memref.dim %arg0, %c2 : memref<?x?x?x?xf32>
//       CHECK:   %[[K:.*]] =  memref.dim %arg0, %c3 : memref<?x?x?x?xf32>
//       CHECK:   %[[B:.*]] =  memref.dim %arg1, %c0 : memref<?x?x?x?xf32>
//       CHECK:   %[[X0:.*]] = memref.dim %arg2, %c1 : memref<?x?x?x?xf32>
//       CHECK:   %[[X1:.*]] = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>
//       CHECK:   scf.for {{.*}} to %[[B]]
//       CHECK:     scf.for {{.*}} to %[[X0]]
//       CHECK:       scf.for {{.*}} to %[[X1]]
//       CHECK:         scf.for {{.*}} to %[[K]]
//       CHECK:           scf.for {{.*}} to %[[Q]]
//       CHECK:             scf.for {{.*}} to %[[Z0]]
//       CHECK:               scf.for {{.*}} to %[[Z1]]
//       CHECK:                 %[[SUM0:.*]] = affine.apply #{{.*}}
//       CHECK:                 %[[SUM1:.*]] = affine.apply #{{.*}}
//       CHECK:                 %[[IDX:.*]] = affine.max #[[$clampMinMap]](%[[SUM0]])
//       CHECK:                 %[[IDY:.*]] = affine.max #[[$clampMinMap]](%[[SUM1]])
//       CHECK:                 memref.load %{{.*}}[%{{.*}}, %[[IDX]], %[[IDY]], %{{.*}}] : memref<?x?x?x?xf32>
//       CHECK:                 select %{{.*}},
//       CHECK:                 memref.load {{.*}} : memref<?x?x?x?xf32>
//       CHECK:                 mulf
//       CHECK:                 memref.load {{.*}} : memref<?x?x?x?xf32>
//       CHECK:                 addf
//       CHECK:                 store %{{.*}}, {{.*}} : memref<?x?x?x?xf32>

// CHECKPARALLEL-LABEL: func @conv_padding
//       CHECKPARALLEL: %{{.*}}: memref<?x?x?x?xf32>, %{{.*}}: memref<?x?x?x?xf32>, %{{.*}}: memref<?x?x?x?xf32>) {
//       CHECKPARALLEL:   %[[ZERO:.*]] = constant 0.000000e+00 : f32
//       CHECKPARALLEL:   %[[Z0:.*]] = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[Z1:.*]] = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[Q:.*]] =  memref.dim %arg0, %c2 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[K:.*]] =  memref.dim %arg0, %c3 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[B:.*]] =  memref.dim %arg1, %c0 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[X0:.*]] = memref.dim %arg2, %c1 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   %[[X1:.*]] = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>
//       CHECKPARALLEL:   scf.parallel (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) to (%[[B]], %[[X0]], %[[X1]], %[[K]]) step (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {
//       CHECKPARALLEL:     scf.for {{.*}} to %[[Q]]
//       CHECKPARALLEL:       scf.for {{.*}} to %[[Z0]]
//       CHECKPARALLEL:         scf.for {{.*}} to %[[Z1]]
//       CHECKPARALLEL:           %[[SUM0:.*]] = affine.apply #{{.*}}
//       CHECKPARALLEL:           %[[SUM1:.*]] = affine.apply #{{.*}}
//       CHECKPARALLEL:           %[[IDX:.*]] = affine.max #[[$clampMinMap]](%[[SUM0]])
//       CHECKPARALLEL:           %[[IDY:.*]] = affine.max #[[$clampMinMap]](%[[SUM1]])
//       CHECKPARALLEL:           memref.load %{{.*}}[%{{.*}}, %[[IDX]], %[[IDY]], %{{.*}}] : memref<?x?x?x?xf32>
//       CHECKPARALLEL:           select %{{.*}},
//       CHECKPARALLEL:           memref.load {{.*}} : memref<?x?x?x?xf32>
//       CHECKPARALLEL:           mulf
//       CHECKPARALLEL:           memref.load {{.*}} : memref<?x?x?x?xf32>
//       CHECKPARALLEL:           addf
//       CHECKPARALLEL:           store %{{.*}}, {{.*}} : memref<?x?x?x?xf32>

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
  linalg.generic #trait2
    ins(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>)
   outs(%arg1, %arg2 : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>,
                       memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %d = mulf %a, %b : f32
      %e = addf %c, %d : f32
      linalg.yield %d, %e : f32, f32
  }
  return
}
// CHECK-LABEL: @generic_region
//       CHECK: scf.for %[[i:.*]] = {{.*}}
//       CHECK:   scf.for %[[j:.*]] = {{.*}}
//       CHECK:     scf.for %[[k:.*]] = {{.*}}
//       CHECK:       %[[a:.*]] = memref.load %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32, #[[$strided2D]]>
//       CHECK:       %[[b:.*]] = memref.load %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:       %[[c:.*]] = memref.load %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:       %[[d:.*]] = mulf %[[a]], %[[b]] : f32
//       CHECK:       %[[e:.*]] = addf %[[c]], %[[d]] : f32
//       CHECK:       store %[[d]], %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:       store %[[e]], %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[$strided3D]]>

// CHECKPARALLEL-LABEL: @generic_region
//       CHECKPARALLEL: scf.parallel (%[[i:[a-zA-Z0-9_]*]], %[[j:[a-zA-Z0-9_]*]], %[[k:[a-zA-Z0-9_]*]])
//       CHECKPARALLEL:   %[[a:.*]] = memref.load %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32, #[[$strided2D]]>
//       CHECKPARALLEL:   %[[b:.*]] = memref.load %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECKPARALLEL:   %[[c:.*]] = memref.load %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[$strided3D]]>
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
func @generic_index_region(
        %arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>,
        %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>,
        %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.generic #trait4
      ins(%arg0 : memref<?x?xf32, offset: ?, strides: [?, 1]>)
     outs(%arg1, %arg2 : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>,
                         memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %k = linalg.index 2 : index
      %result_1 = mulf %a, %b : f32

      %ij = addi %i, %j : index
      %ijk = addi %ij, %k : index
      %ijk_int = index_cast %ijk : index to i32
      %ijk_float = sitofp %ijk_int : i32 to f32

      %result_2 = addf %c, %ijk_float : f32
      linalg.yield %result_1, %result_2 : f32, f32
  }
  return
}

// CHECK-LABEL: @generic_index_region
//       CHECK: scf.for %[[i:.*]] = {{.*}}
//       CHECK:   scf.for %[[j:.*]] = {{.*}}
//       CHECK:     scf.for %[[k:.*]] = {{.*}}
//       CHECK:       %[[a:.*]] = memref.load %{{.*}}[%[[i]], %[[j]]]
//       CHECK:       %[[b:.*]] = memref.load %{{.*}}[%[[i]], %[[j]], %[[k]]]
//       CHECK:       %[[c:.*]] = memref.load %{{.*}}[%[[i]], %[[k]], %[[j]]]
//       CHECK:       %[[result_1:.*]] = mulf %[[a]], %[[b]] : f32
//       CHECK:       %[[ij:.*]] = addi %[[i]], %[[j]] : index
//       CHECK:       %[[ijk:.*]] = addi %[[ij]], %[[k]] : index
//       CHECK:       %[[ijk_int:.*]] = index_cast %[[ijk]] : index to i32
//       CHECK:       %[[ijk_float:.*]] = sitofp %[[ijk_int]] : i32 to f32
//       CHECK:       %[[result_2:.*]] = addf %[[c]], %[[ijk_float]] : f32
//       CHECK:       store %[[result_1]], %{{.*}}[%[[i]], %[[j]], %[[k]]]
//       CHECK:       store %[[result_2]], %{{.*}}[%[[i]], %[[k]], %[[j]]]

// CHECKPARALLEL-LABEL: @generic_index_region
//       CHECKPARALLEL: scf.parallel (%[[i:[a-zA-Z0-9_]*]], %[[j:[a-zA-Z0-9_]*]], %[[k:[a-zA-Z0-9_]*]])
//       CHECKPARALLEL:   %[[a:.*]] = memref.load %{{.*}}[%[[i]], %[[j]]]
//       CHECKPARALLEL:   %[[b:.*]] = memref.load %{{.*}}[%[[i]], %[[j]], %[[k]]]
//       CHECKPARALLEL:   %[[c:.*]] = memref.load %{{.*}}[%[[i]], %[[k]], %[[j]]]
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
  linalg.generic #trait_broadcast
      ins(%arg0 : memref<f32>)
     outs(%arg1 : memref<3x4xf32>) {
    ^bb(%a: f32, %b: f32) :
      linalg.yield %a : f32
  }
  return
}

// CHECK-LABEL: @generic_op_zero_rank
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<3x4xf32>
//       CHECK: scf.for %[[i:.*]] = {{.*}}
//       CHECK:   scf.for %[[j:.*]] = {{.*}}
//       CHECK:     %[[a:.*]] = memref.load %[[ARG0]][]
//       CHECK:     store %[[a]], %[[ARG1]][%[[i]], %[[j]]]

// CHECKPARALLEL-LABEL: @generic_op_zero_rank
//  CHECKPARALLEL-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECKPARALLEL-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<3x4xf32>
//       CHECKPARALLEL: scf.parallel (%[[i:[a-zA-Z0-9_]*]], %[[j:[a-zA-Z0-9_]*]])
//       CHECKPARALLEL:   %[[a:.*]] = memref.load %[[ARG0]][]
//       CHECKPARALLEL:   store %[[a]], %[[ARG1]][%[[i]], %[[j]]]

func @generic_op_scalar(%arg0: f32, %arg1: memref<3x4xf32>)
{
  linalg.generic #trait_broadcast
      ins(%arg0 : f32)
     outs(%arg1 : memref<3x4xf32>) {
    ^bb(%a: f32, %b: f32) :
      linalg.yield %a : f32
  }
  return
}

// CHECK-LABEL: @generic_op_scalar
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: f32
//  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<3x4xf32>
//       CHECK: scf.for %[[i:.*]] = {{.*}}
//       CHECK:   scf.for %[[j:.*]] = {{.*}}
//       CHECK:     store %[[ARG0]], %[[ARG1]][%[[i]], %[[j]]]

// CHECKPARALLEL-LABEL: @generic_op_scalar
//  CHECKPARALLEL-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: f32
//  CHECKPARALLEL-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<3x4xf32>
//       CHECKPARALLEL: scf.parallel (%[[i:[a-zA-Z0-9_]*]], %[[j:[a-zA-Z0-9_]*]])
//       CHECKPARALLEL:   store %[[ARG0]], %[[ARG1]][%[[i]], %[[j]]]

func @generic_index_op_zero_rank(%arg0: memref<i32>, %arg1: memref<3x4xi32>)
{
  linalg.generic #trait_broadcast
      ins(%arg0 : memref<i32>)
     outs(%arg1 : memref<3x4xi32>) {
    ^bb(%a: i32, %b: i32) :
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %ij = addi %i, %j : index
      %ij_int = index_cast %ij : index to i32
      %result = addi %a, %ij_int : i32
      linalg.yield %result : i32
  }
  return
}

// CHECK-LABEL: @generic_index_op_zero_rank
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<i32>
//  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<3x4xi32>
//       CHECK: scf.for %[[i:.*]] = {{.*}}
//       CHECK:   scf.for %[[j:.*]] = {{.*}}
//       CHECK:     %[[a:.*]] = memref.load %[[ARG0]][
//       CHECK:     %[[ij:.*]] = addi %[[i]], %[[j]] : index
//       CHECK:     %[[ij_int:.*]] = index_cast %[[ij]] : index to i32
//       CHECK:     %[[result:.*]] = addi %[[a]], %[[ij_int]] : i32
//       CHECK:     store %[[result]], %[[ARG1]][%[[i]], %[[j]]]

// CHECKPARALLEL-LABEL: @generic_index_op_zero_rank
//  CHECKPARALLEL-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<i32>
//  CHECKPARALLEL-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<3x4xi32>
//       CHECKPARALLEL: scf.parallel (%[[i:[a-zA-Z0-9_]*]], %[[j:[a-zA-Z0-9_]*]])
//       CHECKPARALLEL:   %[[a:.*]] = memref.load %[[ARG0]][
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
  linalg.generic #trait_reduce_1D
      ins(%arg0 : memref<?xf32>)
     outs(%arg1 : memref<f32>) {
    ^bb(%a: f32, %b: f32) :
      %0 = addf %a, %b : f32
      linalg.yield %0 : f32
  }
  return
}
// CHECK-LABEL: @generic_op_1D_reduce
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?xf32>
//  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
//       CHECK: scf.for %[[i:.*]] = {{.*}}
//       CHECK:   %[[a:.*]] = memref.load %[[ARG0]][%[[i]]]
//       CHECK:   %[[b:.*]] = memref.load %[[ARG1]][]
//       CHECK:   %[[c:.*]] = addf %[[a]], %[[b]] : f32
//       CHECK:   store %[[c]], %[[ARG1]][]

// CHECKPARALLEL-LABEL: @generic_op_1D_reduce
//  CHECKPARALLEL-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?xf32>
//  CHECKPARALLEL-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
//       CHECKPARALLEL: scf.for %[[i:.*]] = {{.*}}
//       CHECKPARALLEL:   %[[a:.*]] = memref.load %[[ARG0]][%[[i]]]
//       CHECKPARALLEL:   %[[b:.*]] = memref.load %[[ARG1]][]
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

func @generic_index_op_1D_reduce(%arg0: memref<?xf32>,
                                %arg1: memref<f32>,
                                %arg2: memref<f32>)
{
  linalg.generic #trait_reduce_init_1D
      ins(%arg0, %arg1 : memref<?xf32>, memref<f32>)
     outs(%arg2 : memref<f32>) {
    ^bb(%a: f32, %b: f32, %c: f32) :
      %i = linalg.index 0 : index
      %0 = constant 0 : index
      %1 = cmpi eq, %0, %i : index
      %2 = select %1, %b, %c : f32
      %3 = addf %a, %2 : f32
      linalg.yield %3 : f32
  }
  return
}
// CHECK-LABEL: @generic_index_op_1D_reduce
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?xf32>
//  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<f32>
//       CHECK: scf.for %[[i:.*]] = {{.*}}
//       CHECK:   %[[a:.*]] = memref.load %[[ARG0]][%[[i]]]
//       CHECK:   %[[b:.*]] = memref.load %[[ARG1]][]
//       CHECK:   %[[c:.*]] = memref.load %[[ARG2]][]
//       CHECK:   %[[d:.*]] = select %{{.*}}, %[[b]], %[[c]]
//       CHECK:   %[[e:.*]] = addf %[[a]], %[[d]]
//       CHECK:   store %[[e]], %[[ARG2]][]

// CHECKPARALLEL-LABEL: @generic_index_op_1D_reduce
//  CHECKPARALLEL-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?xf32>
//  CHECKPARALLEL-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECKPARALLEL-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<f32>
//       CHECKPARALLEL: scf.for %[[i:.*]] = {{.*}}
//       CHECKPARALLEL:   %[[a:.*]] = memref.load %[[ARG0]][%[[i]]]
//       CHECKPARALLEL:   %[[b:.*]] = memref.load %[[ARG1]][]
//       CHECKPARALLEL:   %[[c:.*]] = memref.load %[[ARG2]][]
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
  linalg.generic #trait_const_fill outs(%arg0 : memref<?xf32>) {
    ^bb0(%arg1: f32):   // no predecessors
      linalg.yield %cst : f32
    }
    return
}
// CHECK-LABEL: @generic_const_init
//  CHECK-SAME: %[[ARG0:.*]]: memref<?xf32>
//       CHECK: %[[CONST:.*]] = constant 1.000000e+00 : f32
//       CHECK: scf.for %[[i:.*]] = {{.*}}
//       CHECK:   store %[[CONST]], %[[ARG0]]

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
func @scalar_code(%arg0: memref<f32>, %arg1 : memref<f32>, %arg2 : memref<f32>, %arg3 : i1)
{
  linalg.generic #scalar_trait
    ins(%arg0, %arg1 : memref<f32>, memref<f32>)
   outs(%arg2 : memref<f32>) {
  ^bb(%a : f32, %b : f32, %c : f32) :
    %result = scf.if %arg3 -> (f32) {
      scf.yield %a : f32
    } else {
      scf.yield %b : f32
    }
    linalg.yield %result : f32
  }
  return
}
// CHECK-LABEL: @scalar_code
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<f32>
//   CHECK-NOT: scf.for
//       CHECK: memref.load %[[ARG0]][]
//       CHECK: memref.load %[[ARG1]][]
//       CHECK: scf.if
//       CHECK: scf.yield
//       CHECK: else
//       CHECK: scf.yield
//       CHECK: store %{{.*}}, %[[ARG2]][]

// CHECKPARALLEL-LABEL: @scalar_code
//  CHECKPARALLEL-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECKPARALLEL-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
//  CHECKPARALLEL-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<f32>
//   CHECKPARALLEL-NOT: scf.for
//       CHECKPARALLEL: memref.load %[[ARG0]][]
//       CHECKPARALLEL: memref.load %[[ARG1]][]
//       CHECKPARALLEL: scf.if
//       CHECKPARALLEL: scf.yield
//       CHECKPARALLEL: else
//       CHECKPARALLEL: scf.yield
//       CHECKPARALLEL: store %{{.*}}, %[[ARG2]][]

//----------------------------------------------------------------------------//
// Named ops to loops.
//----------------------------------------------------------------------------//
func @named_batch_matmul(%A: memref<?x?x?xf32>, %B: memref<?x?x?xf32>, %C: memref<?x?x?xf32>) {
  linalg.batch_matmul ins(%A, %B : memref<?x?x?xf32>, memref<?x?x?xf32>)
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
//       CHECK: scf.for %[[b:.*]] = %{{.*}} to %[[B]]
//       CHECK:   scf.for %[[m:.*]] = %{{.*}} to %[[M]]
//       CHECK:     scf.for %[[n:.*]] = %{{.*}} to %[[N]]
//       CHECK:       scf.for %[[k:.*]] = %{{.*}} to %[[K]]
//       CHECK:       %[[va:.*]] = memref.load %[[mA]][%[[b]], %[[m]], %[[k]]] : memref<?x?x?xf32>
//       CHECK:       %[[vb:.*]] = memref.load %[[mB]][%[[b]], %[[k]], %[[n]]] : memref<?x?x?xf32>
//       CHECK:       %[[vc:.*]] = memref.load %[[mC]][%[[b]], %[[m]], %[[n]]] : memref<?x?x?xf32>
//       CHECK:       %[[inc:.*]] = mulf %[[va]], %[[vb]] : f32
//       CHECK:       %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECK:       store %[[res]], %[[mC]][%[[b]], %[[m]], %[[n]]] : memref<?x?x?xf32>

// CHECKPARALLEL-LABEL: @named_batch_matmul
//  CHECKPARALLEL-SAME: %[[mA:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECKPARALLEL-SAME: %[[mB:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECKPARALLEL-SAME: %[[mC:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//       CHECKPARALLEL: %[[B:.*]] = memref.dim %[[mA]], %c0 : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[M:.*]] = memref.dim %[[mA]], %c1 : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[K:.*]] = memref.dim %[[mA]], %c2 : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[N:.*]] = memref.dim %[[mB]], %c2 : memref<?x?x?xf32>
//       CHECKPARALLEL: scf.parallel (%[[b:.*]], %[[m:.*]], %[[n:.*]]) = ({{.*}}) to (%[[B]], %[[M]], %[[N]]) step ({{.*}}) {
//       CHECKPARALLEL:   scf.for %[[k:.*]] = %{{.*}} to %[[K]]
//       CHECKPARALLEL:       %[[va:.*]] = memref.load %[[mA]][%[[b]], %[[m]], %[[k]]] : memref<?x?x?xf32>
//       CHECKPARALLEL:       %[[vb:.*]] = memref.load %[[mB]][%[[b]], %[[k]], %[[n]]] : memref<?x?x?xf32>
//       CHECKPARALLEL:       %[[vc:.*]] = memref.load %[[mC]][%[[b]], %[[m]], %[[n]]] : memref<?x?x?xf32>
//       CHECKPARALLEL:       %[[inc:.*]] = mulf %[[va]], %[[vb]] : f32
//       CHECKPARALLEL:       %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECKPARALLEL:       store %[[res]], %[[mC]][%[[b]], %[[m]], %[[n]]] : memref<?x?x?xf32>


func @conv1d_no_symbols(%in : memref<?xf32>, %filter : memref<?xf32>, %out : memref<?xf32>) -> () {
  linalg.conv_1d ins(%in, %filter : memref<?xf32>, memref<?xf32>)
                outs(%out : memref<?xf32>)
  return
}

// CHECK-LABEL: @conv1d_no_symbols
//  CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?xf32>
//  CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?xf32>
//  CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?xf32>
//       CHECK: %[[c0:.*]] = constant 0 : index
//       CHECK: %[[c1:.*]] = constant 1 : index
//       CHECK: %[[dim0:.*]] = memref.dim %[[arg1]], %[[c0]] : memref<?xf32>
//       CHECK: %[[dim1:.*]] = memref.dim %[[arg2]], %[[c0]] : memref<?xf32>
//       CHECK: scf.for %[[b:.*]] = %[[c0]] to %[[dim1]] step %[[c1]] {
//       CHECK:   scf.for %[[m:.*]] = %[[c0]] to %[[dim0]] step %[[c1]] {
//       CHECK:     %[[aff:.*]] = affine.apply #[[$stride1Dilation1]](%[[b]], %[[m]])
//       CHECK:     %[[vb:.*]] = memref.load %[[arg0]][%[[aff]]] : memref<?xf32>
//       CHECK:     %[[va:.*]] = memref.load %[[arg1]][%[[m]]] : memref<?xf32>
//       CHECK:     %[[vc:.*]] = memref.load %[[arg2]][%[[b]]] : memref<?xf32>
//       CHECK:     %[[inc:.*]] = mulf %[[vb]], %[[va]] : f32
//       CHECK:     %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECK:     store %[[res]], %[[arg2]][%[[b]]] : memref<?xf32>

// CHECKPARALLEL-LABEL: @conv1d_no_symbols
//  CHECKPARALLEL-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?xf32>
//  CHECKPARALLEL-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?xf32>
//  CHECKPARALLEL-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?xf32>
//       CHECKPARALLEL: %[[c0:.*]] = constant 0 : index
//       CHECKPARALLEL: %[[c1:.*]] = constant 1 : index
//       CHECKPARALLEL: %[[dim0:.*]] = memref.dim %[[arg1]], %[[c0]] : memref<?xf32>
//       CHECKPARALLEL: %[[dim1:.*]] = memref.dim %[[arg2]], %[[c0]] : memref<?xf32>
//       CHECKPARALLEL: scf.parallel (%[[b:.*]]) = (%[[c0]]) to (%[[dim1]]) step (%[[c1]]) {
//       CHECKPARALLEL:   scf.for %[[m:.*]] = %[[c0]] to %[[dim0]] step %[[c1]] {
//       CHECKPARALLEL:     %[[aff:.*]] = affine.apply #[[$stride1Dilation1]](%[[b]], %[[m]])
//       CHECKPARALLEL:     %[[vb:.*]] = memref.load %[[arg0]][%[[aff]]] : memref<?xf32>
//       CHECKPARALLEL:     %[[va:.*]] = memref.load %[[arg1]][%[[m]]] : memref<?xf32>
//       CHECKPARALLEL:     %[[vc:.*]] = memref.load %[[arg2]][%[[b]]] : memref<?xf32>
//       CHECKPARALLEL:     %[[inc:.*]] = mulf %[[vb]], %[[va]] : f32
//       CHECKPARALLEL:     %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECKPARALLEL:     store %[[res]], %[[arg2]][%[[b]]] : memref<?xf32>


func @conv2d_no_symbols(%in : memref<?x?xf32>, %filter : memref<?x?xf32>, %out : memref<?x?xf32>) -> () {
  linalg.conv_2d ins(%in, %filter : memref<?x?xf32>, memref<?x?xf32>)
                outs(%out: memref<?x?xf32>)
  return
}
// CHECK-LABEL: @conv2d_no_symbols
//  CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?x?xf32>
//  CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?x?xf32>
//  CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?xf32>
//       CHECK: %[[c0:.*]] = constant 0 : index
//       CHECK: %[[c1:.*]] = constant 1 : index
//       CHECK: %[[dim0:.*]] = memref.dim %[[arg1]], %[[c0]] : memref<?x?xf32>
//       CHECK: %[[dim1:.*]] = memref.dim %[[arg1]], %[[c1]] : memref<?x?xf32>
//       CHECK: %[[dim2:.*]] = memref.dim %[[arg2]], %[[c0]] : memref<?x?xf32>
//       CHECK: %[[dim3:.*]] = memref.dim %[[arg2]], %[[c1]] : memref<?x?xf32>
//       CHECK: scf.for %[[arg3:.*]] = %[[c0]] to %[[dim2]] step %[[c1]] {
//       CHECK:   scf.for %[[arg4:.*]] = %[[c0]] to %[[dim3]] step %[[c1]] {
//       CHECK:     scf.for %[[arg5:.*]] = %[[c0]] to %[[dim0]] step %[[c1]] {
//       CHECK:       scf.for %[[arg6:.*]] = %[[c0]] to %[[dim1]] step %[[c1]] {
//       CHECK:         %[[aff:.*]] = affine.apply #[[$stride1Dilation1]](%[[arg3]], %[[arg5]])
//       CHECK:         %[[aff2:.*]] = affine.apply #[[$stride1Dilation1]](%[[arg4]], %[[arg6]])
//       CHECK:         %[[vb:.*]] = memref.load %[[arg0]][%[[aff]], %[[aff2]]] : memref<?x?xf32>

//       CHECK:         %[[va:.*]] = memref.load %[[arg1]][%[[arg5]], %[[arg6]]] : memref<?x?xf32>
//       CHECK:         %[[vc:.*]] = memref.load %[[arg2]][%[[arg3]], %[[arg4]]] : memref<?x?xf32>

//       CHECK:         %[[inc:.*]] = mulf %[[vb]], %[[va]] : f32
//       CHECK:         %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECK:         store %[[res]], %[[arg2]][%[[arg3]], %[[arg4]]] : memref<?x?xf32>

// CHECKPARALLEL-LABEL: @conv2d_no_symbols
//  CHECKPARALLEL-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?x?xf32>
//  CHECKPARALLEL-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?x?xf32>
//  CHECKPARALLEL-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?xf32>
//       CHECKPARALLEL: %[[c0:.*]] = constant 0 : index
//       CHECKPARALLEL: %[[c1:.*]] = constant 1 : index
//       CHECKPARALLEL: %[[dim0:.*]] = memref.dim %[[arg1]], %[[c0]] : memref<?x?xf32>
//       CHECKPARALLEL: %[[dim1:.*]] = memref.dim %[[arg1]], %[[c1]] : memref<?x?xf32>
//       CHECKPARALLEL: %[[dim2:.*]] = memref.dim %[[arg2]], %[[c0]] : memref<?x?xf32>
//       CHECKPARALLEL: %[[dim3:.*]] = memref.dim %[[arg2]], %[[c1]] : memref<?x?xf32>
//       CHECKPARALLEL: scf.parallel (%[[arg3:.*]], %[[arg4:.*]]) = (%[[c0]], %[[c0]]) to (%[[dim2]], %[[dim3]]) step (%[[c1]], %[[c1]]) {
//       CHECKPARALLEL:   scf.for %[[arg5:.*]] = %[[c0]] to %[[dim0]] step %[[c1]] {
//       CHECKPARALLEL:     scf.for %[[arg6:.*]] = %[[c0]] to %[[dim1]] step %[[c1]] {
//       CHECKPARALLEL:       %[[aff:.*]] = affine.apply #[[$stride1Dilation1]](%[[arg3]], %[[arg5]])
//       CHECKPARALLEL:       %[[aff2:.*]] = affine.apply #[[$stride1Dilation1]](%[[arg4]], %[[arg6]])
//       CHECKPARALLEL:       %[[vb:.*]] = memref.load %[[arg0]][%[[aff]], %[[aff2]]] : memref<?x?xf32>
//       CHECKPARALLEL:       %[[va:.*]] = memref.load %[[arg1]][%[[arg5]], %[[arg6]]] : memref<?x?xf32>
//       CHECKPARALLEL:       %[[vc:.*]] = memref.load %[[arg2]][%[[arg3]], %[[arg4]]] : memref<?x?xf32>
//       CHECKPARALLEL:       %[[inc:.*]] = mulf %[[vb]], %[[va]] : f32
//       CHECKPARALLEL:       %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECKPARALLEL:       store %[[res]], %[[arg2]][%[[arg3]], %[[arg4]]] : memref<?x?xf32>


func @conv3d_no_symbols(%in : memref<?x?x?xf32>, %filter : memref<?x?x?xf32>, %out : memref<?x?x?xf32>) -> () {
  linalg.conv_3d ins(%in, %filter : memref<?x?x?xf32>, memref<?x?x?xf32>)
                outs(%out : memref<?x?x?xf32>)
  return
}

// CHECK-LABEL: @conv3d_no_symbols
//  CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//       CHECK-DAG: %[[c2:.*]] = constant 2 : index
//       CHECK-DAG: %[[c0:.*]] = constant 0 : index
//       CHECK-DAG: %[[c1:.*]] = constant 1 : index
//       CHECK: %[[dim0:.*]] = memref.dim %[[arg1]], %[[c0]] : memref<?x?x?xf32>
//       CHECK: %[[dim1:.*]] = memref.dim %[[arg1]], %[[c1]] : memref<?x?x?xf32>
//       CHECK: %[[dim2:.*]] = memref.dim %[[arg1]], %[[c2]] : memref<?x?x?xf32>
//       CHECK: %[[dim3:.*]] = memref.dim %[[arg2]], %[[c0]] : memref<?x?x?xf32>
//       CHECK: %[[dim4:.*]] = memref.dim %[[arg2]], %[[c1]] : memref<?x?x?xf32>
//       CHECK: %[[dim5:.*]] = memref.dim %[[arg2]], %[[c2]] : memref<?x?x?xf32>
//       CHECK: scf.for %[[arg3:.*]] = %[[c0]] to %[[dim3]] step %[[c1]] {
//       CHECK:   scf.for %[[arg4:.*]] = %[[c0]] to %[[dim4]] step %[[c1]] {
//       CHECK:     scf.for %[[arg5:.*]] = %[[c0]] to %[[dim5]] step %[[c1]] {
//       CHECK:       scf.for %[[arg6:.*]] = %[[c0]] to %[[dim0]] step %[[c1]] {
//       CHECK:         scf.for %[[arg7:.*]] = %[[c0]] to %[[dim1]] step %[[c1]] {
//       CHECK:           scf.for %[[arg8:.*]] = %[[c0]] to %[[dim2]] step %[[c1]] {
//       CHECK:             %[[aff:.*]] = affine.apply #[[$stride1Dilation1]](%[[arg3]], %[[arg6]])
//       CHECK:             %[[aff2:.*]] = affine.apply #[[$stride1Dilation1]](%[[arg4]], %[[arg7]])
//       CHECK:             %[[aff3:.*]] = affine.apply #[[$stride1Dilation1]](%[[arg5]], %[[arg8]])
//       CHECK:             %[[vb:.*]] = memref.load %[[arg0]][%[[aff]], %[[aff2]], %[[aff3]]] : memref<?x?x?xf32>

//       CHECK:             %[[va:.*]] = memref.load %[[arg1]][%[[arg6]], %[[arg7]], %[[arg8]]] : memref<?x?x?xf32>
//       CHECK:             %[[vc:.*]] = memref.load %[[arg2]][%[[arg3]], %[[arg4]], %[[arg5]]] : memref<?x?x?xf32>

//       CHECK:             %[[inc:.*]] = mulf %[[vb]], %[[va]] : f32
//       CHECK:             %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECK:             store %[[res]], %[[arg2]][%[[arg3]], %[[arg4]], %[[arg5]]] : memref<?x?x?xf32>

// CHECKPARALLEL-LABEL: @conv3d_no_symbols
//  CHECKPARALLEL-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECKPARALLEL-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//  CHECKPARALLEL-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?x?xf32>
//       CHECKPARALLEL-DAG: %[[c2:.*]] = constant 2 : index
//       CHECKPARALLEL-DAG: %[[c0:.*]] = constant 0 : index
//       CHECKPARALLEL-DAG: %[[c1:.*]] = constant 1 : index
//       CHECKPARALLEL: %[[dim0:.*]] = memref.dim %[[arg1]], %[[c0]] : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[dim1:.*]] = memref.dim %[[arg1]], %[[c1]] : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[dim2:.*]] = memref.dim %[[arg1]], %[[c2]] : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[dim3:.*]] = memref.dim %[[arg2]], %[[c0]] : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[dim4:.*]] = memref.dim %[[arg2]], %[[c1]] : memref<?x?x?xf32>
//       CHECKPARALLEL: %[[dim5:.*]] = memref.dim %[[arg2]], %[[c2]] : memref<?x?x?xf32>
//       CHECKPARALLEL: scf.parallel (%[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]]) = (%[[c0]], %[[c0]], %[[c0]]) to (%[[dim3]], %[[dim4]], %[[dim5]]) step (%[[c1]], %[[c1]], %[[c1]]) {
//       CHECKPARALLEL:   scf.for %[[arg6:.*]] = %[[c0]] to %[[dim0]] step %[[c1]] {
//       CHECKPARALLEL:     scf.for %[[arg7:.*]] = %[[c0]] to %[[dim1]] step %[[c1]] {
//       CHECKPARALLEL:       scf.for %[[arg8:.*]] = %[[c0]] to %[[dim2]] step %[[c1]] {
//       CHECKPARALLEL:         %[[aff:.*]] = affine.apply #[[$stride1Dilation1]](%[[arg3]], %[[arg6]])
//       CHECKPARALLEL:         %[[aff2:.*]] = affine.apply #[[$stride1Dilation1]](%[[arg4]], %[[arg7]])
//       CHECKPARALLEL:         %[[aff3:.*]] = affine.apply #[[$stride1Dilation1]](%[[arg5]], %[[arg8]])
//       CHECKPARALLEL:         %[[vb:.*]] = memref.load %[[arg0]][%[[aff]], %[[aff2]], %[[aff3]]] : memref<?x?x?xf32>
//       CHECKPARALLEL:         %[[va:.*]] = memref.load %[[arg1]][%[[arg6]], %[[arg7]], %[[arg8]]] : memref<?x?x?xf32>
//       CHECKPARALLEL:         %[[vc:.*]] = memref.load %[[arg2]][%[[arg3]], %[[arg4]], %[[arg5]]] : memref<?x?x?xf32>
//       CHECKPARALLEL:         %[[inc:.*]] = mulf %[[vb]], %[[va]] : f32
//       CHECKPARALLEL:         %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECKPARALLEL:         store %[[res]], %[[arg2]][%[[arg3]], %[[arg4]], %[[arg5]]] : memref<?x?x?xf32>

// -----

func @lower_to_loops_with_rank_reducing_subviews(
    %arg0 : memref<?xi32>, %arg1 : memref<?x?xi32>, %arg2 : index,
    %arg3 : index, %arg4 : index) {
  %0 = memref.subview %arg0[%arg2] [%arg3] [1]
      : memref<?xi32> to memref<?xi32, offset: ?, strides: [1]>
  %1 = memref.subview %arg1[0, %arg4] [1, %arg3] [1, 1]
      : memref<?x?xi32> to memref<?xi32, offset: ?, strides : [1]>
  linalg.copy(%0, %1)
      : memref<?xi32, offset: ?, strides: [1]>, memref<?xi32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @lower_to_loops_with_rank_reducing_subviews
//       CHECK:   scf.for %[[IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
//       CHECK:     %[[VAL:.+]] = memref.load %{{.+}}[%[[IV]]]
//       CHECK:     memref.store %[[VAL]], %{{.+}}[%[[IV]]]
//       CHECK:   }

// CHECKPARALLEL-LABEL: func @lower_to_loops_with_rank_reducing_subviews
//       CHECKPARALLEL:   scf.parallel (%[[IV:.+]]) = (%{{.+}}) to (%{{.+}}) step (%{{.+}}) {
//       CHECKPARALLEL:     %[[VAL:.+]] = memref.load %{{.+}}[%[[IV]]]
//       CHECKPARALLEL:     memref.store %[[VAL]], %{{.+}}[%[[IV]]]
//       CHECKPARALLEL:   }
