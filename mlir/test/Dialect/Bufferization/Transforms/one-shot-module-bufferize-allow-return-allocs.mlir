// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs" -split-input-file | FileCheck %s
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs drop-equivalent-func-results=false" -split-input-file | FileCheck %s --check-prefix=EQUIV

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs test-analysis-only analysis-fuzzer-seed=23" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs test-analysis-only analysis-fuzzer-seed=59" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs test-analysis-only analysis-fuzzer-seed=91" -split-input-file -o /dev/null

// Test bufferization using memref types that have no layout map.
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -split-input-file -o /dev/null

// Make sure that the returned buffer is not deallocated.
// TODO: Such buffers currently leak. We need buffer hoisting / ref counting for
// this in the future.

// CHECK-LABEL: func @create_tensor() -> memref<10xf32> {
//       CHECK:   %[[alloc:.*]] = memref.alloc
//       CHECK:   return %[[alloc]]
func.func @create_tensor() -> tensor<10xf32> {
  %0 = linalg.init_tensor [10] : tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK: func @caller(
// CHECK: %[[call:.*]] = call @create_tensor() : () -> memref<10xf32>
// CHECK: %[[extracted:.*]] = memref.load %[[call]]
// CHECK: return %[[extracted]]
func.func @caller(%idx: index) -> f32 {
  %0 = call @create_tensor() : () -> (tensor<10xf32>)
  %1 = tensor.extract %0[%idx] : tensor<10xf32>
  return %1 : f32
}

// -----

// return_slice returns an aliasing tensor. In main, %t is overwritten (but not
// read). This is a conflict because %0 is aliasing with %t. An alloc + copy is
// needed.

// CHECK-LABEL: func @return_slice(
//   CHECK-NOT:   alloc
//   CHECK-NOT:   copy
//       CHECK:   memref.subview
func.func @return_slice(%t: tensor<?xf32>, %sz: index) -> (tensor<?xf32>) {
  %0 = tensor.extract_slice %t[4][%sz][1] : tensor<?xf32> to tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @main(
//  CHECK-SAME:     %[[t:.*]]: memref<?xf32
//       CHECK:   %[[alloc:.*]] = memref.alloc
//   CHECK-DAG:   memref.copy %[[t]], %[[alloc]]
//   CHECK-DAG:   %[[casted:.*]] = memref.cast %[[alloc]]
//       CHECK:   %[[call:.*]] = call @return_slice(%[[casted]]
//       CHECK:   linalg.fill ins({{.*}}) outs(%[[t]]
//       CHECK:   memref.load %[[call]]
//       CHECK:   memref.load %[[t]]
func.func @main(%t: tensor<?xf32>, %sz: index, %idx: index) -> (f32, f32) {
  %cst = arith.constant 1.0 : f32
  %0 = call @return_slice(%t, %sz) : (tensor<?xf32>, index) -> (tensor<?xf32>)
  %filled = linalg.fill ins(%cst : f32) outs(%t : tensor<?xf32>) -> tensor<?xf32>
  %r1 = tensor.extract %0[%idx] : tensor<?xf32>
  %r2 = tensor.extract %filled[%idx] : tensor<?xf32>
  return %r1, %r2 : f32, f32
}

// -----

func.func @return_arg(%A: tensor<?xf32>) -> tensor<?xf32> {
  func.return %A : tensor<?xf32>
}
// CHECK-LABEL: func @return_arg
// CHECK-SAME:      %[[A:.*]]: memref<?xf32
//  CHECK-NOT:    return %[[A]]

// EQUIV-LABEL: func @return_arg
// EQUIV-SAME:      %[[A:.*]]: memref<?xf32
//      EQUIV:    return %[[A]]
