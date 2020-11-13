// RUN: mlir-opt -test-finalizing-bufferize -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @void_function_signature_conversion
func @void_function_signature_conversion(%arg0: tensor<4x8xf32>) {
    return
}
// CHECK: ({{.*}}: memref<4x8xf32>)

// -----

// CHECK-LABEL: func @complex_signature_conversion
func @complex_signature_conversion(
  %arg0: tensor<5xf32>,
  %arg1: memref<10xf32>,
  %arg2: i1,
  %arg3: f16) -> (
    i1,
    tensor<5xf32>,
    memref<10xf32>,
    memref<15xf32>,
    f16) {
  %0 = alloc() : memref<15xf32>
  %1 = test.tensor_based in(%arg0 : tensor<5xf32>) -> tensor<5xf32>
  return %arg2, %1, %arg1, %0, %arg3 :
   i1, tensor<5xf32>, memref<10xf32>, memref<15xf32>, f16
}
//      CHECK: (%[[ARG0:.*]]: memref<5xf32>, %[[ARG1:.*]]: memref<10xf32>,
// CHECK-SAME: %[[ARG2:.*]]: i1, %[[ARG3:.*]]: f16)
// CHECK-SAME: (i1, memref<5xf32>, memref<10xf32>, memref<15xf32>, f16)
//      CHECK: %[[FIRST_ALLOC:.*]] = alloc()
//      CHECK: %[[TENSOR_ALLOC:.*]] = alloc()
//      CHECK: return %[[ARG2]], %[[TENSOR_ALLOC]], %[[ARG1]], %[[FIRST_ALLOC]],
// CHECK-SAME: %[[ARG3]]

// -----

// CHECK-LABEL: func @no_signature_conversion_is_needed
func @no_signature_conversion_is_needed(%arg0: memref<4x8xf32>) {
  return
}
// CHECK: ({{.*}}: memref<4x8xf32>)

// -----

// CHECK-LABEL: func @no_signature_conversion_is_needed
func @no_signature_conversion_is_needed(%arg0: i1, %arg1: f16) -> (i1, f16){
  return %arg0, %arg1 : i1, f16
}
// CHECK: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: f16) -> (i1, f16)
// CHECK: return %[[ARG0]], %[[ARG1]]

// -----

// CHECK-LABEL: func @simple_signature_conversion
func @simple_signature_conversion(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  return %arg0 : tensor<4x8xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[TYPE:.*]]<[[RANK:.*]]>) -> [[TYPE]]<[[RANK]]>
// CHECK-NEXT: return %[[ARG0]]

// -----

// CHECK-LABEL: func @func_with_unranked_arg_and_result
func @func_with_unranked_arg_and_result(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  return %arg0 : tensor<*xf32>
}
// CHECK-SAME: ([[ARG:%.*]]: memref<*xf32>) -> memref<*xf32>
// CHECK-NEXT: return [[ARG]] : memref<*xf32>

// -----

// CHECK-LABEL: func @func_and_block_signature_conversion
func @func_and_block_signature_conversion(%arg0 : tensor<2xf32>, %cond : i1, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32>{
    cond_br %cond, ^bb1, ^bb2
  ^bb1:
    br ^exit(%arg0 : tensor<2xf32>)
  ^bb2:
    br ^exit(%arg0 : tensor<2xf32>)
  ^exit(%arg2: tensor<2xf32>):
    return %arg1 : tensor<4x4xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[ARG0_TYPE:.*]], %[[COND:.*]]: i1, %[[ARG1:.*]]: [[ARG1_TYPE:.*]]) -> [[RESULT_TYPE:.*]] {
//      CHECK: br ^[[EXIT_BLOCK:.*]](%[[ARG0]] : [[ARG0_TYPE]])
//      CHECK: br ^[[EXIT_BLOCK]](%[[ARG0]] : [[ARG0_TYPE]])
//      CHECK: ^[[EXIT_BLOCK]](%{{.*}}: [[ARG0_TYPE]])
// CHECK-NEXT:  return %[[ARG1]] : [[RESULT_TYPE]]

// -----

// CHECK-LABEL: func @callee
func @callee(%arg1: tensor<5xf32>) -> (tensor<5xf32>, memref<2xf32>) {
  %buff = alloc() : memref<2xf32>
  return %arg1, %buff : tensor<5xf32>, memref<2xf32>
}
// CHECK: (%[[CALLEE_ARG:.*]]: memref<5xf32>) -> (memref<5xf32>, memref<2xf32>)
// CHECK: %[[ALLOC:.*]] = alloc()
// CHECK: return %[[CALLEE_ARG]], %[[ALLOC]]

// CHECK-LABEL: func @caller
func @caller(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %x:2 = call @callee(%arg0) : (tensor<5xf32>) -> (tensor<5xf32>, memref<2xf32>)
  %y:2 = call @callee(%x#0) : (tensor<5xf32>) -> (tensor<5xf32>, memref<2xf32>)
  return %y#0 : tensor<5xf32>
}
// CHECK: (%[[CALLER_ARG:.*]]: memref<5xf32>) -> memref<5xf32>
// CHECK: %[[X:.*]]:2 = call @callee(%[[CALLER_ARG]])
// CHECK: %[[Y:.*]]:2 = call @callee(%[[X]]#0)
// CHECK: return %[[Y]]#0

// -----

// Test case: Testing BufferizeCallOpConverter to see if it matches with the
// signature of the new signature of the callee function when there are tuple
// typed args and results. BufferizeTypeConverter is set to flatten tuple typed
// arguments. The tuple typed values should be decomposed and composed using
// get_tuple_element and make_tuple operations of test dialect. Tensor types are
// converted to Memref. Memref typed function results remain as function
// results.

// CHECK-LABEL: func @callee
func @callee(%arg0: tuple<tensor<2xf32>,i1, tensor<5xf32>>) -> (tuple<tensor<2xf32>,i1, tensor<5xf32>>){
  return %arg0 : tuple<tensor<2xf32>,i1, tensor<5xf32>>
}
// CHECK-SAME: (%[[ARG0:.*]]: memref<2xf32>, %[[ARG1:.*]]: i1, %[[ARG2:.*]]: memref<5xf32>)
// CHECK-SAME: (memref<2xf32>, i1, memref<5xf32>)
// CHECK-NEXT: %[[TUPLE:.*]] = "test.make_tuple"(%[[ARG0]], %[[ARG1]], %[[ARG2]])
// CHECK-NEXT: %[[FIRST_ELEM:.*]] = "test.get_tuple_element"(%[[TUPLE]]) {index = 0 : i32}
// CHECK-NEXT: %[[SECOND_ELEM:.*]] = "test.get_tuple_element"(%[[TUPLE]]) {index = 1 : i32}
// CHECK-NEXT: %[[THIRD_ELEM:.*]]  = "test.get_tuple_element"(%[[TUPLE]]) {index = 2 : i32}
// CHECK-NEXT: return %[[FIRST_ELEM]], %[[SECOND_ELEM]], %[[THIRD_ELEM]]

// CHECK-LABEL: func @caller
func @caller(%arg0: tuple<tensor<2xf32>,i1, tensor<5xf32>>) -> tuple<tensor<2xf32>,i1, tensor<5xf32>>{
  %x0 = call @callee(%arg0) : (tuple<tensor<2xf32>,i1, tensor<5xf32>>) -> (tuple<tensor<2xf32>,i1, tensor<5xf32>>)
  %y0 = call @callee(%x0) : (tuple<tensor<2xf32>,i1, tensor<5xf32>>) -> (tuple<tensor<2xf32>,i1, tensor<5xf32>>)
  return %y0 : tuple<tensor<2xf32>,i1, tensor<5xf32>>
}
// CHECK-SAME: (%[[ARG0:.*]]: memref<2xf32>, %[[ARG1:.*]]: i1, %[[ARG2:.*]]: memref<5xf32>)
// CHECK-SAME: (memref<2xf32>, i1, memref<5xf32>)
// CHECK-NEXT: %[[ARG_TUPLE:.*]] = "test.make_tuple"(%[[ARG0]], %[[ARG1]], %[[ARG2]])
// CHECK-NEXT: %[[FIRST_ELEM:.*]] = "test.get_tuple_element"(%[[ARG_TUPLE]]) {index = 0 : i32}
// CHECK-NEXT: %[[SECOND_ELEM:.*]] = "test.get_tuple_element"(%[[ARG_TUPLE]]) {index = 1 : i32}
// CHECK-NEXT: %[[THIRD_ELEM:.*]]  = "test.get_tuple_element"(%[[ARG_TUPLE]]) {index = 2 : i32}
// CHECK-NEXT: %[[CALLEE_RESULTS:.*]]:3 = call @callee(%[[FIRST_ELEM]], %[[SECOND_ELEM]], %[[THIRD_ELEM]])
// CHECK-SAME: (memref<2xf32>, i1, memref<5xf32>) -> (memref<2xf32>, i1, memref<5xf32>)
// CHECK-NEXT: %[[RESULT_TUPLE:.*]] = "test.make_tuple"(%[[CALLEE_RESULTS]]#0, %[[CALLEE_RESULTS]]#1, %[[CALLEE_RESULTS]]#2)
// CHECK-NEXT: %[[FIRST_ELEM:.*]] = "test.get_tuple_element"(%[[RESULT_TUPLE]]) {index = 0 : i32}
// CHECK-NEXT: %[[SECOND_ELEM:.*]] = "test.get_tuple_element"(%[[RESULT_TUPLE]]) {index = 1 : i32}
// CHECK-NEXT: %[[THIRD_ELEM:.*]]  = "test.get_tuple_element"(%[[RESULT_TUPLE]]) {index = 2 : i32}
// CHECK-NEXT: %[[CALLEE_RESULTS:.*]]:3 = call @callee(%[[FIRST_ELEM]], %[[SECOND_ELEM]], %[[THIRD_ELEM]])
// CHECK-SAME: (memref<2xf32>, i1, memref<5xf32>) -> (memref<2xf32>, i1, memref<5xf32>)
// CHECK-NEXT: %[[RETURN_TUPLE:.*]] = "test.make_tuple"(%[[CALLEE_RESULTS]]#0, %[[CALLEE_RESULTS]]#1, %[[CALLEE_RESULTS]]#2)
// CHECK-NEXT: %[[FIRST_ELEM:.*]] = "test.get_tuple_element"(%[[RETURN_TUPLE]]) {index = 0 : i32}
// CHECK-NEXT: %[[SECOND_ELEM:.*]] = "test.get_tuple_element"(%[[RETURN_TUPLE]]) {index = 1 : i32}
// CHECK-NEXT: %[[THIRD_ELEM:.*]]  = "test.get_tuple_element"(%[[RETURN_TUPLE]]) {index = 2 : i32}
// CHECK-NEXT: return %[[FIRST_ELEM]], %[[SECOND_ELEM]], %[[THIRD_ELEM]]

// -----

// Test case: Testing BufferizeFuncOpConverter and
// BufferizeReturnOpConverter to see if the return operation matches with the
// new function signature when there are tuple typed args and results.
// BufferizeTypeConverter is set to flatten tuple typed arguments. The tuple
// typed values should be decomposed and composed using get_tuple_element and
// make_tuple operations of test dialect. Tensor types are converted to Memref.
// Memref typed function results remain as function results.

// CHECK-LABEL: func @decompose_tuple_typed_function_args_and_results
func @decompose_tuple_typed_function_args_and_results(%arg0: tuple<i1,f32>, %arg1: tensor<10xf32>, %arg2: tuple<i1, tensor<5xf32>>) -> (tuple<i1, tensor<5xf32>>, tensor<10xf32>, tuple<i1,f32>){
  return %arg2, %arg1, %arg0 : tuple<i1, tensor<5xf32>>, tensor<10xf32>, tuple<i1,f32>
}
// CHECK-SAME: %[[ARG0:.*]]: i1, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: memref<10xf32>, %[[ARG3:.*]]: i1, %[[ARG4:.*]]: memref<5xf32>
// CHECK-SAME: (i1, memref<5xf32>, memref<10xf32>, i1, f32)
// CHECK-NEXT: %[[FIRST_TUPLE:.*]] = "test.make_tuple"(%[[ARG0]], %[[ARG1]])
// CHECK-NEXT: %[[SECOND_TUPLE:.*]] = "test.make_tuple"(%[[ARG3]], %[[ARG4]])
// CHECK-NEXT: %[[SECOND_TUPLE_FIRST_ELEM:.*]]  = "test.get_tuple_element"(%[[SECOND_TUPLE]]) {index = 0 : i32}
// CHECK-NEXT: %[[SECOND_TUPLE_SECOND_ELEM:.*]] = "test.get_tuple_element"(%[[SECOND_TUPLE]]) {index = 1 : i32}
// CHECK-NEXT: %[[FIRST_TUPLE_FIRST_ELEM:.*]] = "test.get_tuple_element"(%[[FIRST_TUPLE]]) {index = 0 : i32}
// CHECK-NEXT: %[[FIRST_TUPLE_SECOND_ELEM:.*]] = "test.get_tuple_element"(%[[FIRST_TUPLE]]) {index = 1 : i32}
// CHECK-NEXT: return %[[SECOND_TUPLE_FIRST_ELEM]], %[[SECOND_TUPLE_SECOND_ELEM]], %[[ARG2]], %[[FIRST_TUPLE_FIRST_ELEM]], %[[FIRST_TUPLE_SECOND_ELEM]]
