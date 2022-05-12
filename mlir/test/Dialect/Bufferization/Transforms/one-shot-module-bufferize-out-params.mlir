// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs promote-buffer-results-to-out-params" -split-input-file | FileCheck %s

// Note: This bufferization is not very efficient yet, but it works.

// CHECK: #[[$map1:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
// CHECK-LABEL: func @callee(
//  CHECK-SAME:              %[[arg0:.*]]: memref<5xf32, #[[$map1]]>,
//  CHECK-SAME:              %[[arg1:.*]]: memref<5xf32>) {
//       CHECK:   %[[alloc:.*]] = memref.alloc() {{.*}} : memref<5xf32>
//       CHECK:   memref.copy %[[arg0]], %[[alloc]]
//       CHECK:   memref.store %{{.*}}, %[[alloc]]
//       CHECK:   memref.copy %[[alloc]], %[[arg1]]
//       CHECK:   memref.dealloc %[[alloc]]
//       CHECK:   return
//       CHECK: }
func.func @callee(%t: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 8.0 : f32
  %1 = tensor.insert %cst into %t[%c0] : tensor<5xf32>
  return %t, %1 : tensor<5xf32>, tensor<5xf32>
}

// CHECK: func @main(%[[arg0:.*]]: memref<5xf32, #[[$map1]]>) -> (f32, f32) {
// CHECK:   %[[alloc:.*]] = memref.alloc() : memref<5xf32>
// CHECK:   call @callee(%[[arg0]], %[[alloc]])
// CHECK:   %[[l1:.*]] = memref.load %[[arg0]]
// CHECK:   %[[l2:.*]] = memref.load %[[alloc]]
// CHECK:   memref.dealloc %[[alloc]]
// CHECK:   return %[[l1]], %[[l2]]
// CHECK: }
func.func @main(%t: tensor<5xf32>) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %0, %1 = func.call @callee(%t)
      : (tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)
  %2 = tensor.extract %0[%c0] : tensor<5xf32>
  %3 = tensor.extract %1[%c0] : tensor<5xf32>
  return %2, %3 : f32, f32
}

