// RUN: mlir-opt %s -inline | FileCheck %s

// CHECK-LABEL: func @test_inline
// CHECK-SAME: (%[[ARG:.*]]: memref<*xf32>)
// CHECK-NOT: call
// CHECK: %[[RES:.*]] = bufferization.clone %[[ARG]]
// CHECK: return %[[RES]]
func.func @test_inline(%buf : memref<*xf32>) -> memref<*xf32> {
  %0 = call @inner_func(%buf) : (memref<*xf32>) -> memref<*xf32>
  return %0 : memref<*xf32>
}

func.func @inner_func(%buf : memref<*xf32>) -> memref<*xf32> {
  %clone = bufferization.clone %buf : memref<*xf32> to memref<*xf32>
  return %clone : memref<*xf32>
}
