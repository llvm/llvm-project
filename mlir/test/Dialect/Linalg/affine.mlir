// RUN: mlir-opt %s -convert-linalg-to-affine-loops | FileCheck %s

// Test that we can lower all the way to LLVM without crashing, don't check results here.
// RUN: mlir-opt %s -convert-linalg-to-affine-loops -convert-linalg-to-llvm -o=/dev/null 2>&1

// CHECK-DAG: #[[$strided3D:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>

// CHECK-DAG: #[[$stride2Dilation1:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>

// CHECK-DAG: #[[$clampMinMap:.*]] = affine_map<(d0) -> (d0, 0)>

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
// CHECK-SAME: [[M:arg[0-9]+]]: index
// CHECK-SAME: [[N:arg[0-9]+]]: index
// CHECK-SAME: [[K:arg[0-9]+]]: index
//       CHECK: %[[A:.*]] = memref.view %{{.*}} : memref<?xi8> to memref<?x?xf32>
//       CHECK: %[[B:.*]] = memref.view %{{.*}} : memref<?xi8> to memref<?x?xf32>
//       CHECK: %[[C:.*]] = memref.view %{{.*}} : memref<?xi8> to memref<?x?xf32>
//       CHECK: affine.for
//       CHECK:   affine.for
//       CHECK:     affine.for
//   CHECK-DAG:       %[[a:.*]] = affine.load %[[A]]{{.*}} : memref<?x?xf32>
//   CHECK-DAG:       %[[b:.*]] = affine.load %[[B]]{{.*}} : memref<?x?xf32>
//   CHECK-DAG:       %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:       %[[c:.*]] = affine.load %[[C]]{{.*}} : memref<?x?xf32>
//   CHECK-DAG:       %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:       affine.store %[[res]], %[[C]]{{.*}} : memref<?x?xf32>

func @conv_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {strides = [2]}: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}

// CHECK-LABEL: func @conv_view3(
//  CHECK: %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>) {
//       CHECK:   %[[Z0:.*]] = memref.dim %arg0, %c0 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:   %[[Q:.*]] = memref.dim %arg0, %c1 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:   %[[K:.*]] = memref.dim %arg0, %c2 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:   %[[B:.*]] = memref.dim %arg1, %c0 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:   %[[X0:.*]] = memref.dim %arg2, %c1 : memref<?x?x?xf32, #[[$strided3D]]>
//       CHECK:   affine.for {{.*}}0 to %[[B]] {
//       CHECK:     affine.for {{.*}}0 to %[[X0]] {
//       CHECK:       affine.for {{.*}}0 to %[[K]] {
//       CHECK:         affine.for {{.*}}0 to %[[Q]] {
//       CHECK:           affine.for {{.*}}0 to %[[Z0]] {
//       CHECK:            %[[SUM:.*]] = affine.apply #[[$stride2Dilation1]]{{.*}}
//       No padding needed here; only affine loads.
//       CHECK-NEXT:       affine.load
//       CHECK-NEXT:       affine.load

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
//       CHECK:   affine.for {{.*}}0 to %[[B]] {
//       CHECK:     affine.for {{.*}}0 to %[[X0]] {
//       CHECK:       affine.for {{.*}}0 to %[[X1]] {
//       CHECK:         affine.for {{.*}}0 to %[[K]] {
//       CHECK:           affine.for {{.*}}0 to %[[Q]] {
//       CHECK:             affine.for {{.*}}0 to %[[Z0]] {
//       CHECK:               affine.for {{.*}}0 to %[[Z1]] {
//       CHECK:                 %[[SUM0:.*]] = affine.apply #{{.*}}
//       CHECK:                 %[[SUM1:.*]] = affine.apply #{{.*}}
//       CHECK:                 %[[IDX:.*]] = affine.max #[[$clampMinMap]](%[[SUM0]])
//       CHECK:                 %[[IDY:.*]] = affine.max #[[$clampMinMap]](%[[SUM1]])
// Padded conv involves an affine.max in the memory access and this is not
// allowed by affine.load. Use memref.load in such cases.
//       CHECK:                 memref.load %{{.*}}[%{{.*}}, %[[IDX]], %[[IDY]], %{{.*}}] : memref<?x?x?x?xf32>
//       CHECK:                 select {{.*}} : f32
//       CHECK:                 affine.load
//       CHECK:                 mulf {{.*}} : f32
//       CHECK:                 affine.load
//       CHECK:                 addf {{.*}} : f32
//       CHECK:                 affine.store

//----------------------------------------------------------------------------//
// Named ops to loops.
//----------------------------------------------------------------------------//
func @named_batch_matmul(%A: memref<?x?x?xf32>, %B: memref<?x?x?xf32>, %C: memref<?x?x?xf32>) {
  linalg.batch_matmul ins(%A, %B: memref<?x?x?xf32>, memref<?x?x?xf32>)
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
//       CHECK: affine.for %[[b:.*]] = {{.*}}0 to %[[B]] {
//       CHECK:   affine.for %[[m:.*]] = {{.*}}0 to %[[M]] {
//       CHECK:     affine.for %[[n:.*]] = {{.*}}0 to %[[N]] {
//       CHECK:       affine.for %[[k:.*]] = {{.*}}0 to %[[K]] {
//       CHECK:       %[[va:.*]] = affine.load %[[mA]][%[[b]], %[[m]], %[[k]]] : memref<?x?x?xf32>
//       CHECK:       %[[vb:.*]] = affine.load %[[mB]][%[[b]], %[[k]], %[[n]]] : memref<?x?x?xf32>
//       CHECK:       %[[vc:.*]] = affine.load %[[mC]][%[[b]], %[[m]], %[[n]]] : memref<?x?x?xf32>
//       CHECK:       %[[inc:.*]] = mulf %[[va]], %[[vb]] : f32
//       CHECK:       %[[res:.*]] = addf %[[vc]], %[[inc]] : f32
//       CHECK:       affine.store %[[res]], %[[mC]][%[[b]], %[[m]], %[[n]]] : memref<?x?x?xf32>

