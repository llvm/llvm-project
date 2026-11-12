// RUN: mlir-opt %s --sparsification-and-bufferization | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/177856:
// sparse_tensor.foreach over a rank-0 (scalar) dense tensor must not crash.
// The LoopEmitter called getValPosits() which invoked std::vector::back()
// on an empty container because no loop levels were entered for rank-0.

// CHECK-LABEL: func.func @foreach_scalar_no_reduc(
// CHECK-SAME:    %[[A:.*]]: memref<i32>)
// CHECK-NOT:   memref.load
// CHECK:       return
func.func @foreach_scalar_no_reduc(%arg0: tensor<i32>) {
  sparse_tensor.foreach in %arg0 : tensor<i32> do {
    ^bb0(%v: i32):
  }
  return
}

// CHECK-LABEL: func.func @foreach_scalar_with_reduc(
// CHECK-SAME:    %[[A:.*]]: memref<i32>
// CHECK-SAME:    %[[B:.*]]: i32)
// CHECK:         %[[VAL:.*]] = memref.load %[[A]][] : memref<i32>
// CHECK:         return %[[VAL]] : i32
func.func @foreach_scalar_with_reduc(%arg0: tensor<i32>, %arg1: i32) -> i32 {
  %ret = sparse_tensor.foreach in %arg0 init(%arg1): tensor<i32>, i32 -> i32
  do {
    ^bb0(%v: i32, %r: i32):
      sparse_tensor.yield %v : i32
  }
  return %ret : i32
}
