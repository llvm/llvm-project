// RUN: mlir-opt -split-input-file -shape-tensor-to-memref <%s | FileCheck %s

// -----
// Check that shape.assuming returns a memref.
//
// CHECK-LABEL: @shape_assuming_returns_memref
func @shape_assuming_returns_memref() {
  %0 = shape.const_witness true
  // CHECK: shape.assuming %{{.*}} -> (memref<2xf16>) {
  %1 = shape.assuming %0 -> (tensor<2xf16>) {
    %2 = "test.source"() : () -> (tensor<2xf16>)
    shape.assuming_yield %2 : tensor<2xf16>
  }
  "test.sink"(%1) : (tensor<2xf16>) -> ()
  return
}


