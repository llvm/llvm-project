// RUN: mlir-opt %s -inline | FileCheck %s

// CHECK-LABEL: func @test_inline
// CHECK-SAME: (%[[ARG:.*]]: memref<*xf32>)
// CHECK-NOT: call
// CHECK: %[[RES:.*]] = bufferization.clone %[[ARG]]
// CHECK: bufferization.dealloc
// CHECK: return %[[RES]]
func.func @test_inline(%buf : memref<*xf32>) -> (memref<*xf32>, i1) {
  %0:2 = call @inner_func(%buf) : (memref<*xf32>) -> (memref<*xf32>, i1)
  return %0#0, %0#1 : memref<*xf32>, i1
}

func.func @inner_func(%buf : memref<*xf32>) -> (memref<*xf32>, i1) {
  %true = arith.constant true
  %clone = bufferization.clone %buf : memref<*xf32> to memref<*xf32>
  %0 = bufferization.dealloc (%buf : memref<*xf32>) if (%true) retain (%clone : memref<*xf32>)
  return %clone, %0 : memref<*xf32>, i1
}
