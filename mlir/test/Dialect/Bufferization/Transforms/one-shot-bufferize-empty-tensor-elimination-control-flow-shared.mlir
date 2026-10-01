// RUN: mlir-opt %s -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries" -cse -canonicalize -split-input-file | FileCheck %s

//      CHECK: func @shared_empty(
// CHECK-SAME: %[[FUNC_ARG_1:[0-9a-zA-Z]*]]: i1
// CHECK-SAME: %[[FUNC_ARG_2:[0-9a-zA-Z]*]]: f32
// CHECK-SAME: %[[FUNC_ARG_3:[0-9a-zA-Z]*]]: f32
// CHECK-SAME: %[[FUNC_ARG_4:[0-9a-zA-Z]*]]: memref<16xf32, strided{{.*}}>
func.func @shared_empty(
    %arg0: i1,
    %arg1: f32,
    %arg2: f32,
    %arg3: tensor<16xf32>)
        -> tensor<16xf32>
{
    // CHECK-NOT: memref.alloc
    // CHECK: %[[subview:.*]] = memref.subview %[[FUNC_ARG_4]][4] [4] [1]
    %0 = tensor.empty() : tensor<4xf32>
    // CHECK: scf.if %[[FUNC_ARG_1]] {
    %1 = scf.if %arg0 -> (tensor<4xf32>) {
        // CHECK: linalg.fill ins(%[[FUNC_ARG_2]]{{.*}}) outs(%[[subview]] : memref<4xf32, strided{{.*}}>)
        %2 = linalg.fill ins(%arg1 : f32) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
        scf.yield %2 : tensor<4xf32>
        // CHECK: } else {
    } else {
        // CHECK: linalg.fill ins(%[[FUNC_ARG_3]]{{.*}}) outs(%[[subview]] : memref<4xf32, strided{{.*}}>)
        %2 = linalg.fill ins(%arg2 : f32) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
        scf.yield %2 : tensor<4xf32>
    }
    // CHECK-NOT: memref.copy
    // CHECK:   return %[[FUNC_ARG_4]]
    %inserted_slice = tensor.insert_slice %1 into %arg3[4] [4] [1] : tensor<4xf32> into tensor<16xf32>
    return %inserted_slice : tensor<16xf32>
}