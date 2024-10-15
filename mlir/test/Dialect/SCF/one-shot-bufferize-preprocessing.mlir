// RUN: mlir-opt %s -scf-loop-bufferization-preprocessing -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -canonicalize | FileCheck %s

// CHECK-LABEL: func @conflict_in_loop(
//  CHECK-SAME:     %[[A:.*]]: memref<10xf32>
func.func @conflict_in_loop(%A: tensor<10xf32>, %f: f32, %idx: index, %lb: index, %ub: index, %step: index) -> f32 {
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
  %r = scf.for %i = %lb to %ub step %step iter_args(%tA = %A) -> (tensor<10xf32>) {
    // CHECK: %[[alloc:.*]] = memref.alloc()
    // CHECK: memref.copy %[[A]], %[[alloc]]
    // CHECK: memref.store %{{.*}}, %[[alloc]]
    %0 = tensor.insert %f into %tA[%i] : tensor<10xf32>
    // CHECK: %[[read:.*]] = memref.load %[[A]]
    %read = tensor.extract %tA[%idx] : tensor<10xf32>
    // CHECK: vector.print %[[read]]
    vector.print %read : f32
    // CHECK: memref.copy %[[alloc]], %[[A]]
    scf.yield %0 : tensor<10xf32>
  }

  // CHECK: memref.load %[[A]]
  %f0 = tensor.extract %r[%step] : tensor<10xf32>
  return %f0 : f32
}
