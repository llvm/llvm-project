// RUN: mlir-opt %s -convert-linalg-to-loops | FileCheck %s

// Test that we can lower all the way to LLVM without crashing, don't check results here.
// RUN: mlir-opt %s --convert-linalg-to-llvm -o=/dev/null 2>&1

// CHECK-DAG: #[[strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG: #[[strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[strided3D:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>
// CHECK-DAG: #[[strided4D:.*]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
// CHECK-DAG: #[[clampMinMap:.*]] = affine_map<(d0) -> (d0, 0)>

// CHECK-DAG: #[[Stride2Dilation1:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK-DAG: #[[Stride2Dilation4:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1 * 4)>
// CHECK-DAG: #[[Stride3Dilation5:.*]] = affine_map<(d0, d1) -> (d0 * 3 + d1 * 5)>


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
//       CHECK: loop.for %{{.*}} = %{{.*}} to %[[M]] step %{{.*}} {
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECK-DAG:       %[[a:.*]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:       %[[b:.*]] = load %[[B]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:       %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:       %[[c:.*]] = load %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:       %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:       store %[[res]], %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>

func @matvec(%arg0: memref<?xi8>, %M: index, %N: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %2 = view %arg0[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %3 = view %arg0[%c0][%M] : memref<?xi8> to memref<?xf32, offset: ?, strides: [1]>
  %4 = view %arg0[%c0][%N] : memref<?xi8> to memref<?xf32, offset: ?, strides: [1]>
  linalg.matvec(%2, %3, %4) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @matvec(%{{.*}}: memref<?xi8>,
// CHECK-SAME: [[M:arg[0-9]+]]: index
// CHECK-SAME: [[K:arg[0-9]+]]: index
//       CHECK: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECK: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECK: loop.for %{{.*}} = %{{.*}} to %[[M]] step %{{.*}} {
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECK-DAG:     %[[a:.*]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:     %[[b:.*]] = load %[[B]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECK-DAG:     %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:     %[[c:.*]] = load %[[C]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECK-DAG:     %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:     store %[[res]], %[[C]][%{{.*}}] : memref<?xf32, #[[strided1D]]>

func @dot(%arg0: memref<?xi8>, %M: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %1 = view %arg0[%c0][%M] : memref<?xi8> to memref<?xf32, offset: ?, strides: [1]>
  %2 = view %arg0[%c0][%M] : memref<?xi8> to memref<?xf32, offset: ?, strides: [1]>
  %3 = view %arg0[][] : memref<?xi8> to memref<f32>
  linalg.dot(%1, %2, %3) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>, memref<f32>
  return
}
// CHECK-LABEL: func @dot(%{{.*}}: memref<?xi8>,
// CHECK-SAME: [[K:arg[0-9]+]]: index
//       CHECK: %[[A:.*]] = std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECK: %[[B:.*]] = std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECK: %[[C:.*]] = std.view %{{.*}}[][] : memref<?xi8> to memref<f32>
//       CHECK: loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECK-DAG:   %[[a:.*]] = load %[[A]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECK-DAG:   %[[b:.*]] = load %[[B]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECK-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:   %[[c:.*]] = load %[[C]][] : memref<f32>
//   CHECK-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:   store %[[res]], %[[C]][] : memref<f32>

func @dot_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>, memref<f32>
  return
}
// CHECK-LABEL: func @dot_view(
//       CHECK:   %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<f32>) {
//       CHECK: %[[K:.*]] = dim %arg0, 0 : memref<?xf32, #[[strided1D]]>
//       CHECK: loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECK-DAG:   %[[a:.*]] = load %arg0[%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECK-DAG:   %[[b:.*]] = load %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECK-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:   %[[c:.*]] = load %{{.*}}[] : memref<f32>
//   CHECK-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:   store %[[res]], %{{.*}}[] : memref<f32>

func @fill_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<?xf32, offset: ?, strides: [1]>, f32
  return
}
// CHECK-LABEL: func @fill_view(
//       CHECK: %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: f32) {
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>

func @fill_view0(%arg0: memref<f32>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<f32>, f32
  return
}
// CHECK-LABEL: func @fill_view0(%{{.*}}: memref<f32>, %{{.*}}: f32) {
//       CHECK:   store %{{.*}}, %{{.*}}[] : memref<f32>

func @fill_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, f32
  return
}
// CHECK-LABEL: func @fill_view3(
//       CHECK: %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: f32) {
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:         store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>

func @copy_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.copy(%arg0, %arg1) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @copy_view(
//       CHECK: %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<?xf32, #[[strided1D]]>) {
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     %[[L:.*]] = load %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>
//       CHECK:     store %[[L]], %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>

func @copy_view0(%arg0: memref<f32>, %arg1: memref<f32>) {
  linalg.copy(%arg0, %arg1) : memref<f32>, memref<f32>
  return
}
// CHECK-LABEL: func @copy_view0(%{{.*}}: memref<f32>, %{{.*}}: memref<f32>) {
//       CHECK:   %{{.*}} = load %{{.*}}[] : memref<f32>
//       CHECK:   store %{{.*}}, %{{.*}}[] : memref<f32>

func @copy_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.copy(%arg0, %arg1) {inputPermutation = affine_map<(i, j, k) -> (i, k, j)>,
                             outputPermutation = affine_map<(i, j, k) -> (k, j, i)>} :
    memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @copy_view3
//       CHECK: (%{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>) {
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:         %[[L:.*]] = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:         store %[[L]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>

func @conv_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {strides = [2]}: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @conv_view3(
//       CHECK: %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>) {
//       CHECK:   %[[Z0:.*]] = dim %arg0, 0 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   %[[Q:.*]] = dim %arg0, 1 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   %[[K:.*]] = dim %arg0, 2 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   %[[B:.*]] = dim %arg1, 0 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   %[[X0:.*]] = dim %arg2, 1 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECK:         loop.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECK:           loop.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECK:             %[[SUM:.*]] = affine.apply #[[Stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECK:             %{{.*}} = load %{{.*}}[%{{.*}}, %[[SUM]], %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:             %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECK:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:             %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECK:             store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>

func @conv_view4(%arg0: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg1: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg2: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [4, 5], strides = [2, 3]} : memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>
  return
}
// CHECK-LABEL: func @conv_view4(
//       CHECK: %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>) {
//       CHECK:   %[[Z0:.*]] = dim %arg0, 0 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:   %[[Z1:.*]] = dim %arg0, 1 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:   %[[Q:.*]] = dim %arg0, 2 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:   %[[K:.*]] = dim %arg0, 3 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:   %[[B:.*]] = dim %arg1, 0 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:   %[[X0:.*]] = dim %arg2, 1 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:   %[[X1:.*]] = dim %arg2, 2 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %[[X1]] step %{{.*}} {
//       CHECK:         loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECK:           loop.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECK:             loop.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECK:               loop.for %{{.*}} = %{{.*}} to %[[Z1]] step %{{.*}} {
//       CHECK:                 %[[SUM0:.*]] = affine.apply #[[Stride2Dilation4]](%{{.*}}, %{{.*}})
//       CHECK:                 %[[SUM1:.*]] = affine.apply #[[Stride3Dilation5]](%{{.*}}, %{{.*}})
//       CHECK:                 %{{.*}} = load %{{.*}}[%{{.*}}, %[[SUM0]], %[[SUM1]], %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:                 %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:                 %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECK:                 %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:                 %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECK:                 store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>

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
//       CHECK:   %[[Z0:.*]] = dim %arg0, 0 : memref<?x?x?x?xf32>
//       CHECK:   %[[Z1:.*]] = dim %arg0, 1 : memref<?x?x?x?xf32>
//       CHECK:   %[[Q:.*]] =  dim %arg0, 2 : memref<?x?x?x?xf32>
//       CHECK:   %[[K:.*]] =  dim %arg0, 3 : memref<?x?x?x?xf32>
//       CHECK:   %[[B:.*]] =  dim %arg1, 0 : memref<?x?x?x?xf32>
//       CHECK:   %[[X0:.*]] = dim %arg2, 1 : memref<?x?x?x?xf32>
//       CHECK:   %[[X1:.*]] = dim %arg2, 2 : memref<?x?x?x?xf32>
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %[[X1]] step %{{.*}} {
//       CHECK:         loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECK:           loop.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECK:             loop.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECK:               loop.for %{{.*}} = %{{.*}} to %[[Z1]] step %{{.*}} {
//       CHECK:                 %[[SUM0:.*]] = affine.apply #{{.*}}(%{{.*}}, %{{.*}})
//       CHECK:                 %[[SUM1:.*]] = affine.apply #{{.*}}(%{{.*}}, %{{.*}})
//       CHECK:                 %[[IDX:.*]] = affine.max #[[clampMinMap]](%[[SUM0]])
//       CHECK:                 %[[IDY:.*]] = affine.max #[[clampMinMap]](%[[SUM1]])
//       CHECK:                 %{{.*}} = load %{{.*}}[%{{.*}}, %[[IDX]], %[[IDY]], %{{.*}}] : memref<?x?x?x?xf32>
//       CHECK:                 %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : f32
//       CHECK:                 %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32>
//       CHECK:                 %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECK:                 %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32>
//       CHECK:                 %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECK:                 store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32>

func @foo(%0: f32, %1: f32, %2: f32) -> (f32, f32) {
  %f0 = constant 0.0 : f32
  return %f0, %f0 : f32, f32
}
#accesses = [
  affine_map<(i, j, k) -> (i, j)>,
  affine_map<(i, j, k) -> (i, j, k)>,
  affine_map<(i, j, k) -> (i, k, j)>
]
#trait = {
  args_in = 1,
  args_out = 2,
  iterator_types = ["parallel", "parallel", "parallel"],
  indexing_maps = #accesses,
  fun = @foo,
  library_call = "some_external_function_name_1",
  doc = "B(i,j,k), C(i,k,j) = foo(A(i, j), B(i,j,k), C(i,k,j))"
}
func @generic_function(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.generic #trait %arg0, %arg1, %arg2:
    memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: @foo
// CHECK-LABEL: @generic_function
//       CHECK: loop.for %[[i:.*]] = {{.*}}
//       CHECK:   loop.for %[[j:.*]] = {{.*}}
//       CHECK:     loop.for %[[k:.*]] = {{.*}}
//       CHECK:       %[[a:.*]] = load %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32, #[[strided2D]]>
//       CHECK:       %[[b:.*]] = load %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:       %[[c:.*]] = load %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:       %[[res:.*]]:2 = call @foo(%[[a]], %[[b]], %[[c]]) : (f32, f32, f32) -> (f32, f32)
//       CHECK:       store %[[res]]#0, %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:       store %[[res]]#1, %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[strided3D]]>

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
// CHECK-LABEL: @generic_region
//       CHECK: loop.for %[[i:.*]] = {{.*}}
//       CHECK:   loop.for %[[j:.*]] = {{.*}}
//       CHECK:     loop.for %[[k:.*]] = {{.*}}
//       CHECK:       %[[a:.*]] = load %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32, #[[strided2D]]>
//       CHECK:       %[[b:.*]] = load %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:       %[[c:.*]] = load %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:       %[[d:.*]] = mulf %[[a]], %[[b]] : f32
//       CHECK:       %[[e:.*]] = addf %[[c]], %[[d]] : f32
//       CHECK:       store %[[d]], %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:       store %[[e]], %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[strided3D]]>

func @indexed_foo(%i: index, %j: index, %k: index, %0: f32, %1: f32, %2: f32) -> (f32, f32) {
  %i_int = index_cast %i: index to i32
  %i_float = sitofp %i_int : i32 to f32
  return %i_float, %i_float : f32, f32
}
#trait3 = {
  args_in = 1,
  args_out = 2,
  iterator_types = ["parallel", "parallel", "parallel"],
  indexing_maps = #accesses,
  fun = @indexed_foo,
  library_call = "some_external_function_name_1",
  doc = "b(i,j,k), c(i,k,j) = foo(a(i, j), b(i,j,k), c(i,k,j))"
}
func @indexed_generic_function(
         %arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>,
         %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>,
         %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.indexed_generic #trait3 %arg0, %arg1, %arg2:
    memref<?x?xf32, offset: ?, strides: [?, 1]>,
    memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>,
    memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: @indexed_foo
// CHECK-LABEL: @indexed_generic_function
// CHECK: loop.for %[[i:.*]] = {{.*}}
// CHECK:   loop.for %[[j:.*]] = {{.*}}
// CHECK:     loop.for %[[k:.*]] = {{.*}}
// CHECK:       %[[a:.*]] = load %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32, #[[strided2D]]>
// CHECK:       %[[b:.*]] = load %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[strided3D]]>
// CHECK:       %[[c:.*]] = load %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[strided3D]]>
// CHECK:       %[[res:.*]]:2 = call @indexed_foo(%[[i]], %[[j]], %[[k]], %[[a]], %[[b]], %[[c]]) : (index, index, index, f32, f32, f32) -> (f32, f32)
// CHECK:       store %[[res]]#0, %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[strided3D]]>
// CHECK:       store %[[res]]#1, %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[strided3D]]>

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

// CHECK-LABEL: @indexed_generic_region
// CHECK: loop.for %[[i:.*]] = {{.*}}
// CHECK:   loop.for %[[j:.*]] = {{.*}}
// CHECK:     loop.for %[[k:.*]] = {{.*}}
// CHECK:       %[[a:.*]] = load %{{.*}}[%[[i]], %[[j]]]
// CHECK:       %[[b:.*]] = load %{{.*}}[%[[i]], %[[j]], %[[k]]]
// CHECK:       %[[c:.*]] = load %{{.*}}[%[[i]], %[[k]], %[[j]]]
// CHECK:       %[[result_1:.*]] = mulf %[[a]], %[[b]] : f32
// CHECK:       %[[ij:.*]] = addi %[[i]], %[[j]] : index
// CHECK:       %[[ijk:.*]] = addi %[[ij]], %[[k]] : index
// CHECK:       %[[ijk_int:.*]] = index_cast %[[ijk]] : index to i32
// CHECK:       %[[ijk_float:.*]] = sitofp %[[ijk_int]] : i32 to f32
// CHECK:       %[[result_2:.*]] = addf %[[c]], %[[ijk_float]] : f32
// CHECK:       store %[[result_1]], %{{.*}}[%[[i]], %[[j]], %[[k]]]
// CHECK:       store %[[result_2]], %{{.*}}[%[[i]], %[[k]], %[[j]]]

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

// CHECK-LABEL: @generic_op_zero_rank
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<f32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<3x4xf32>
// CHECK: loop.for %[[i:.*]] = {{.*}}
// CHECK:   loop.for %[[j:.*]] = {{.*}}
// CHECK:     %[[a:.*]] = load %[[ARG0]][]
// CHECK:     store %[[a]], %[[ARG1]][%[[i]], %[[j]]]

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

// CHECK-LABEL: @indexed_generic_op_zero_rank
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<i32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<3x4xi32>
// CHECK: loop.for %[[i:.*]] = {{.*}}
// CHECK:   loop.for %[[j:.*]] = {{.*}}
// CHECK:     %[[a:.*]] = load %[[ARG0]][
// CHECK:     %[[ij:.*]] = addi %[[i]], %[[j]] : index
// CHECK:     %[[ij_int:.*]] = index_cast %[[ij]] : index to i32
// CHECK:     %[[result:.*]] = addi %[[a]], %[[ij_int]] : i32
// CHECK:     store %[[result]], %[[ARG1]][%[[i]], %[[j]]]

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
// CHECK-LABEL: @generic_op_1D_reduce
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
// CHECK: loop.for %[[i:.*]] = {{.*}}
// CHECK:   %[[a:.*]] = load %[[ARG0]][%[[i]]]
// CHECK:   %[[b:.*]] = load %[[ARG1]][]
// CHECK:   %[[c:.*]] = addf %[[a]], %[[b]] : f32
// CHECK:   store %[[c]], %[[ARG1]][]


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
// CHECK-LABEL: @indexed_generic_op_1D_reduce
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<f32>
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<f32>
// CHECK: loop.for %[[i:.*]] = {{.*}}
// CHECK:   %[[a:.*]] = load %[[ARG0]][%[[i]]]
// CHECK:   %[[b:.*]] = load %[[ARG1]][]
// CHECK:   %[[c:.*]] = load %[[ARG2]][]
// CHECK:   %[[d:.*]] = select %{{.*}}, %[[b]], %[[c]]
// CHECK:   %[[e:.*]] = addf %[[a]], %[[d]]
// CHECK:   store %[[e]], %[[ARG2]][]
