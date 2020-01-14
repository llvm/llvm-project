// RUN: mlir-opt %s -convert-linalg-to-affine-loops | FileCheck %s

// Test that we can lower all the way to LLVM without crashing, don't check results here.
// RUN: mlir-opt %s --convert-linalg-to-llvm -o=/dev/null 2>&1

// CHECK-DAG: #[[strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[strided3D:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>

// CHECK-DAG: #[[stride2Dilation1:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>

func @matmul(%arg0: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %A = view %arg0[%c0][%M, %K] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %B = view %arg0[%c0][%K, %N] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %C = view %arg0[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}

// CHECK-LABEL: func @matmul(%{{.*}}: memref<?xi8>,
// CHECK-SAME: [[M:arg[0-9]+]]: index
// CHECK-SAME: [[N:arg[0-9]+]]: index
// CHECK-SAME: [[K:arg[0-9]+]]: index
//       CHECK: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECK: affine.for %{{.*}}  = 0 to %{{.*}} {
//       CHECK:   affine.for %{{.*}} = 0 to %{{.*}} {
//       CHECK:     affine.for %{{.*}} = 0 to %{{.*}} {
//   CHECK-DAG:       %[[a:.*]] = affine.load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:       %[[b:.*]] = affine.load %[[B]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:       %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:       %[[c:.*]] = affine.load %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:       %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:       affine.store %[[res]], %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>

func @conv_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {strides = [2]}: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}

// CHECK-LABEL: func @conv_view3(
//  CHECK: %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>) {
//       CHECK:   %[[Z0:.*]] = dim %arg0, 0 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   %[[Q:.*]] = dim %arg0, 1 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   %[[K:.*]] = dim %arg0, 2 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   %[[B:.*]] = dim %arg1, 0 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   %[[X0:.*]] = dim %arg2, 1 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   affine.for %{{.*}} = 0 to %[[B]] {
//       CHECK:     affine.for %{{.*}} = 0 to %[[X0]] {
//       CHECK:       affine.for %{{.*}} = 0 to %[[K]] {
//       CHECK:         affine.for %{{.*}} = 0 to %[[Q]] {
//       CHECK:           affine.for %{{.*}} = 0 to %[[Z0]] {
//       CHECK:            %[[SUM:.*]] = affine.apply #[[stride2Dilation1]](%{{.*}}, %{{.*}})
