// RUN: mlir-opt -p 'builtin.module(buffer-results-to-out-params{add-result-attr})' -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @basic({{.*}}: memref<f32> {bufferize.result})
func.func @basic() -> (memref<f32>) {
  %0 = "test.source"() : () -> (memref<f32>)
  return %0 : memref<f32>
}

// -----

// CHECK-LABEL: multiple_results
// CHECK-SAME:  memref<1xf32> {bufferize.result}
// CHECK-SAME:  memref<2xf32> {bufferize.result}
func.func @multiple_results() -> (memref<1xf32>, memref<2xf32>) {
  %0, %1 = "test.source"() : () -> (memref<1xf32>, memref<2xf32>)
  return %0, %1 : memref<1xf32>, memref<2xf32>
}
