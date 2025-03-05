// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 no-analysis-func-filter=contains_to_memref_op" -drop-equivalent-buffer-results --split-input-file | FileCheck %s

// ToMemref ops do not pass analysis step. CopyBeforeWrite will be true only for the
// FuncOp "contains_to_memref_op" since it is specified in no-analysis-func-filter.

// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 copy-before-write=1" -drop-equivalent-buffer-results --split-input-file | FileCheck %s --check-prefix=CHECK_COPY

// Show that memref.copy appear in both functions when CopyBeforeWrite is true.

module {
  // CHECK-LABEL:   func.func @foo(
  // CHECK-NOT:       memref.copy

  // CHECK_COPY-LABEL:   func.func @foo(
  // CHECK_COPY:           memref.copy

  func.func @foo(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %inserted = tensor.insert %cst into %arg0[%c0] : tensor<?xf32>
    return %inserted : tensor<?xf32>
  }

  // CHECK-LABEL:   func.func @contains_to_memref_op(
  // CHECK:           memref.copy

  // CHECK_COPY-LABEL:   func.func @contains_to_memref_op(
  // CHECK_COPY:           memref.copy

  func.func @contains_to_memref_op(%arg0: tensor<?xf32> {bufferization.writable = true}, %arg1: index) -> vector<5xf32> {
    %0 = bufferization.to_memref %arg0 : tensor<?xf32> to memref<?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = vector.transfer_read %0[%arg1], %cst : memref<?xf32>, vector<5xf32>
    return %1 : vector<5xf32>
  }
}
