// RUN: mlir-opt -split-input-file --one-shot-bufferize="dialect-filter=shape,bufferization copy-before-write unknown-type-conversion=identity-layout-map allow-unknown-ops" <%s | FileCheck %s

// -----

// CHECK-LABEL:   func @shape_assuming() {
// CHECK:           %[[WTRUE:.*]] = shape.const_witness true
// CHECK:           %[[MEMREF:.*]] = shape.assuming %[[WTRUE]] -> (memref<2xf16>) {
// CHECK:             %[[TENSOR_VAL:.*]] = "test.source"() : () -> tensor<2xf16>
// CHECK:             %[[YIELDED_MEMREF:.*]] = bufferization.to_memref %[[TENSOR_VAL]] : memref<2xf16>
// CHECK:             shape.assuming_yield %[[YIELDED_MEMREF]] : memref<2xf16>
// CHECK:           }
// CHECK:           %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF:.*]] : memref<2xf16>
// CHECK:           "test.sink"(%[[TENSOR]]) : (tensor<2xf16>) -> ()
// CHECK:           return
// CHECK:         }
func.func @shape_assuming() {
  %0 = shape.const_witness true
  %1 = shape.assuming %0 -> (tensor<2xf16>) {
    %2 = "test.source"() : () -> (tensor<2xf16>)
    shape.assuming_yield %2 : tensor<2xf16>
  }
  "test.sink"(%1) : (tensor<2xf16>) -> ()
  return
}
