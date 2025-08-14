// RUN: mlir-opt %s -allow-unregistered-dialect -pass-pipeline='builtin.module(test.symbol_scope_isolated(test-one-shot-module-bufferize))' -split-input-file | FileCheck %s

"test.symbol_scope_isolated"() ({
  // CHECK-LABEL: func @inner_func(
  //  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
  func.func @inner_func(%t: tensor<?xf32>) -> (tensor<?xf32>, f32) {
    // CHECK-NOT: copy
    %f = arith.constant 1.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: memref.store %{{.*}}, %[[arg0]]
    %0 = tensor.insert %f into %t[%c0] : tensor<?xf32>
    // CHECK: %[[load:.*]] = memref.load %[[arg0]]
    %1 = tensor.extract %0[%c1] : tensor<?xf32>
    // CHECK: return %[[arg0]], %[[load]] : memref<?xf32{{.*}}>, f32
    return %0, %1 : tensor<?xf32>, f32
  }

  // CHECK-LABEL: func @call_func_with_non_tensor_return(
  //  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
  func.func @call_func_with_non_tensor_return(
      %t0: tensor<?xf32> {bufferization.writable = true}) -> (f32, tensor<?xf32>) {
    // CHECK-NOT: alloc
    // CHECK-NOT: copy
    // CHECK: %[[call:.*]]:2 = call @inner_func(%[[arg0]])
    %0, %1 = call @inner_func(%t0) : (tensor<?xf32>) -> (tensor<?xf32>, f32)
    // CHECK: return %[[call]]#1, %[[call]]#0 : f32, memref<?xf32,{{.*}}>
    return %1, %0 : f32, tensor<?xf32>
  }
  "test.finish" () : () -> ()
}) : () -> ()


