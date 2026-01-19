// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries" | FileCheck %s
// XFAIL: *

module {
  func.func @foo(%arg0: tensor<64x20x40xf32>) -> tensor<64x20x40xf32> {
    %u = tensor.cast %arg0 : tensor<64x20x40xf32> to tensor<*xf32>
    %r = call @relu(%u) : (tensor<*xf32>) -> tensor<*xf32>
    %b = tensor.cast %r : tensor<*xf32> to tensor<64x20x40xf32>
    return %b : tensor<64x20x40xf32>
  }
  func.func private @relu(tensor<*xf32>) -> tensor<*xf32>
}

// CHECK-LABEL: func.func @foo
// CHECK-SAME: -> memref<64x20x40xf32
// CHECK: %[[R:.*]] = call @relu
// CHECK: %[[C:.*]] = memref.cast %[[R]] : memref<*xf32> to memref<64x20x40xf32
// CHECK: return %[[C]] : memref<64x20x40xf32

