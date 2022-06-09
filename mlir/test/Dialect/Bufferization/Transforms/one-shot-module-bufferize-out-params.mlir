// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs promote-buffer-results-to-out-params function-boundary-type-conversion=fully-dynamic-layout-map" -buffer-deallocation -split-input-file | FileCheck %s
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs promote-buffer-results-to-out-params function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -split-input-file | FileCheck %s --check-prefix=CHECK-NO-LAYOUT
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=infer-layout-map" -buffer-deallocation -split-input-file | FileCheck %s --check-prefix=CHECK-BASELINE

// Note: function-boundary-type-conversion=infer-layout-map with
// promote-buffer-results-to-out-params is an unsupported combination.

// Note: This bufferization is not very efficient yet, but it works.

// CHECK: #[[$map1:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
// CHECK-LABEL: func @callee(
//  CHECK-SAME:              %[[arg0:.*]]: memref<5xf32, #[[$map1]]>,
//  CHECK-SAME:              %[[arg1:.*]]: memref<5xf32, #[[$map1]]>) {
// This alloc is not needed, but it is inserted due to the out-of-place
// bufferization of the tensor.insert. With a better layering of the out param
// promotion pass, this alloc could be avoided.
//       CHECK:   %[[alloc:.*]] = memref.alloc() {{.*}} : memref<5xf32>
//       CHECK:   memref.copy %[[arg0]], %[[alloc]]
//       CHECK:   memref.store %{{.*}}, %[[alloc]]
//       CHECK:   memref.copy %[[alloc]], %[[arg1]]
//       CHECK:   memref.dealloc %[[alloc]]
//       CHECK:   return
//       CHECK: }

// CHECK-NO-LAYOUT-LABEL: func @callee(
//  CHECK-NO-LAYOUT-SAME:     %[[arg0:.*]]: memref<5xf32>,
//  CHECK-NO-LAYOUT-SAME:     %[[arg1:.*]]: memref<5xf32>) {
//       CHECK-NO-LAYOUT:   %[[alloc:.*]] = memref.alloc() {{.*}} : memref<5xf32>
//       CHECK-NO-LAYOUT:   memref.copy %[[arg0]], %[[alloc]]
//       CHECK-NO-LAYOUT:   memref.store {{.*}}, %[[alloc]]
//       CHECK-NO-LAYOUT:   memref.copy %[[alloc]], %[[arg1]]
//       CHECK-NO-LAYOUT:   memref.dealloc %[[alloc]]

// CHECK-BASELINE: #[[$map1:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
// CHECK-BASELINE-LABEL: func @callee(
//  CHECK-BASELINE-SAME:     %[[arg0:.*]]: memref<5xf32, #[[$map1]]>) -> memref<5xf32> {
//       CHECK-BASELINE:   %[[alloc:.*]] = memref.alloc() {{.*}} : memref<5xf32>
//       CHECK-BASELINE:   memref.copy %[[arg0]], %[[alloc]]
//       CHECK-BASELINE:   memref.store {{.*}}, %[[alloc]]
//       CHECK-BASELINE:   return %[[alloc]]
func.func @callee(%t: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 8.0 : f32
  // This must bufferize out-of-place.
  %1 = tensor.insert %cst into %t[%c0] : tensor<5xf32>
  // Instead of returning %1, copy into new out param. %t will disappear
  // entirely because the buffer is equivalent to a bbArg.
  return %t, %1 : tensor<5xf32>, tensor<5xf32>
}

// CHECK: func @main(%[[arg0:.*]]: memref<5xf32, #[[$map1]]>) -> (f32, f32) {
// CHECK:   %[[alloc:.*]] = memref.alloc() : memref<5xf32>
// CHECK:   %[[casted:.*]] = memref.cast %[[alloc]] : memref<5xf32> to memref<5xf32, #[[$map1]]>
// CHECK:   call @callee(%[[arg0]], %[[casted]])
// CHECK:   %[[l1:.*]] = memref.load %[[arg0]]
// CHECK:   %[[l2:.*]] = memref.load %[[alloc]]
// CHECK:   memref.dealloc %[[alloc]]
// CHECK:   return %[[l1]], %[[l2]]
// CHECK: }

// CHECK-NO-LAYOUT-LABEL: func @main(%{{.*}}: memref<5xf32>) -> (f32, f32) {
//       CHECK-NO-LAYOUT:   %[[alloc:.*]] = memref.alloc() : memref<5xf32>
//       CHECK-NO-LAYOUT:   call @callee(%{{.*}}, %[[alloc]])
func.func @main(%t: tensor<5xf32>) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %0, %1 = func.call @callee(%t)
      : (tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)
  %2 = tensor.extract %0[%c0] : tensor<5xf32>
  %3 = tensor.extract %1[%c0] : tensor<5xf32>
  return %2, %3 : f32, f32
}

// -----

// CHECK: #[[$map2a:.*]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
// CHECK: #[[$map2b:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 20 + s0 + d1)>
// CHECK-LABEL: func @callee(
//  CHECK-SAME:     %{{.*}}: index,
//  CHECK-SAME:     %[[r:.*]]: memref<2x5xf32, #[[$map2a]]>) {
//       CHECK:   %[[alloc:.*]] = memref.alloc() {{.*}} : memref<10x20xf32>
//       CHECK:   %[[subview:.*]] = memref.subview %[[alloc]]{{.*}} : memref<10x20xf32> to memref<2x5xf32, #[[$map2b]]>
//       CHECK:   memref.copy %[[subview]], %[[r]]
//       CHECK:   memref.dealloc %[[alloc]]

// CHECK-NO-LAYOUT-LABEL: func @callee(
//  CHECK-NO-LAYOUT-SAME:              %{{.*}}: index,
//  CHECK-NO-LAYOUT-SAME:              %[[r:.*]]: memref<2x5xf32>) {
//       CHECK-NO-LAYOUT:   %[[alloc:.*]] = memref.alloc() {{.*}} : memref<10x20xf32>
//       CHECK-NO-LAYOUT:   %[[subview:.*]] = memref.subview %[[alloc]]
// Note: This alloc is not needed, but it is inserted before the returned buffer
// is promoted to an out param to reconcile mismatching layout maps on return
// value and function signature.
//       CHECK-NO-LAYOUT:   %[[alloc2:.*]] = memref.alloc() : memref<2x5xf32>
//       CHECK-NO-LAYOUT:   memref.copy %[[subview]], %[[alloc2]]
//       CHECK-NO-LAYOUT:   memref.dealloc %[[alloc]]
//       CHECK-NO-LAYOUT:   memref.copy %[[alloc2]], %[[r]]
//       CHECK-NO-LAYOUT:   memref.dealloc %[[alloc2]]

// CHECK-BASELINE: #[[$map2:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 20 + s0 + d1)>
// CHECK-BASELINE-LABEL: func @callee(
//  CHECK-BASELINE-SAME:     %{{.*}}: index) -> memref<2x5xf32, #[[$map2]]> {
//       CHECK-BASELINE:   %[[alloc:.*]] = memref.alloc() {{.*}} : memref<10x20xf32>
//       CHECK-BASELINE:   %[[subview:.*]] = memref.subview %[[alloc]]
//       CHECK-BASELINE:   return %[[subview]]
func.func @callee(%idx: index) -> tensor<2x5xf32> {
  %0 = bufferization.alloc_tensor() : tensor<10x20xf32>
  %1 = tensor.extract_slice %0[%idx, %idx][2, 5][1, 1] : tensor<10x20xf32> to tensor<2x5xf32>
  return %1 : tensor<2x5xf32>
}

// CHECK: func @main(
// CHECK:   %[[alloc:.*]] = memref.alloc() : memref<2x5xf32>
// CHECK:   %[[casted:.*]] = memref.cast %[[alloc]] : memref<2x5xf32> to memref<2x5xf32, #[[$map2a]]>
// CHECK:   call @callee(%{{.*}}, %[[casted]])
// CHECK:   memref.load %[[alloc]]
// CHECK:   memref.dealloc %[[alloc]]

// CHECK-NO-LAYOUT: func @main(
// CHECK-NO-LAYOUT:   %[[alloc:.*]] = memref.alloc() : memref<2x5xf32>
// CHECK-NO-LAYOUT:   call @callee(%{{.*}}, %[[alloc]])
// CHECK-NO-LAYOUT:   memref.load %[[alloc]]
// CHECK-NO-LAYOUT:   memref.dealloc

// CHECK-BASELINE: func @main(
// CHECK-BASELINE:   %[[call:.*]] = call @callee
// CHECK-BASELINE:   memref.load %[[call]]
func.func @main(%idx: index) -> f32 {
  %c0 = arith.constant 0 : index
  %0 = func.call @callee(%idx) : (index) -> (tensor<2x5xf32>)
  %1 = tensor.extract %0[%c0, %c0] : tensor<2x5xf32>
  return %1 : f32
}
