// RUN: mlir-opt -test-buffer-placement-preparation -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @func_signature_conversion
func @func_signature_conversion(%arg0: tensor<4x8xf32>) {
    return
}
// CHECK: ({{.*}}: memref<4x8xf32>) {

// -----

// Only tensor typed function result should be converted to memref and move to the
// function arguments list. The other memref function results remain as function
// results.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @memref_in_function_results
func @memref_in_function_results(%arg0: tensor<5xf32>, %arg1: memref<10xf32>) -> (tensor<5xf32>, memref<10xf32>, memref<15xf32>) {
  %0 = alloc() : memref<15xf32>
  %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
    ins(%arg0 : tensor<5xf32>) {
    ^bb0(%gen1_arg0: f32):
      %tmp1 = exp %gen1_arg0 : f32
      linalg.yield %tmp1 : f32
    } -> tensor<5xf32>
  return %1, %arg1, %0 : tensor<5xf32>, memref<10xf32>, memref<15xf32>
}
//      CHECK: (%[[ARG0:.*]]: memref<5xf32>, %[[ARG1:.*]]: memref<10xf32>, %[[RESULT:.*]]: memref<5xf32>)
// CHECK-SAME: (memref<10xf32>, memref<15xf32>)
//      CHECK: %[[FIRST_ALLOC:.*]] = alloc()
//      CHECK: %[[LINALG_ALLOC:.*]] = alloc()
//      CHECK: linalg.copy(%[[LINALG_ALLOC]], %[[RESULT]])
//      CHECK: return %[[ARG1]], %[[FIRST_ALLOC]]

// -----

// CHECK-LABEL: func @no_signature_conversion_is_needed
func @no_signature_conversion_is_needed(%arg0: memref<4x8xf32>) {
  return
}
// CHECK: ({{.*}}: memref<4x8xf32>) {

// -----

// CHECK-LABEL: func @no_signature_conversion_is_needed
func @no_signature_conversion_is_needed(%arg0: i1, %arg1: f16) -> (i1, f16){
  return %arg0, %arg1 : i1, f16
}
// CHECK: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: f16) -> (i1, f16)
// CHECK: return %[[ARG0]], %[[ARG1]]

// -----

// CHECK-LABEL: func @complex_signature_conversion
func @complex_signature_conversion(%arg0: tensor<4x8xf32>, %arg1: i1, %arg2: tensor<5x5xf64>,%arg3: f16) -> (i1, tensor<5x5xf64>, f16, tensor<4x8xf32>) {
    return %arg1, %arg2, %arg3, %arg0 : i1, tensor<5x5xf64>, f16, tensor<4x8xf32>
}
//      CHECK: (%[[ARG0:.*]]: memref<4x8xf32>, %[[ARG1:.*]]: i1, %[[ARG2:.*]]: memref<5x5xf64>, %[[ARG3:.*]]: f16,
// CHECK-SAME: %[[RESULT1:.*]]: memref<5x5xf64>, %[[RESULT2:.*]]: memref<4x8xf32>) -> (i1, f16) {
// CHECK-NEXT: linalg.copy(%[[ARG2]], %[[RESULT1]])
// CHECK-NEXT: linalg.copy(%[[ARG0]], %[[RESULT2]])
// CHECK-NEXT: return %[[ARG1]], %[[ARG3]]

// -----

// CHECK-LABEL: func @non_void_to_void_return_op_converter
func @non_void_to_void_return_op_converter(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  return %arg0 : tensor<4x8xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[TYPE:.*]]<[[RANK:.*]]>, %[[RESULT:.*]]: [[TYPE]]<[[RANK]]>) {
// CHECK-NEXT: linalg.copy(%[[ARG0]], %[[RESULT]])
// CHECK-NEXT: return

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
//      CHECK: (%[[ARG0:.*]]: [[ARG0_TYPE:.*]], %[[COND:.*]]: i1, %[[ARG1:.*]]: [[ARG1_TYPE:.*]], %[[RESULT:.*]]: [[RESULT_TYPE:.*]]) {
//      CHECK: br ^[[EXIT_BLOCK:.*]](%[[ARG0]] : [[ARG0_TYPE]])
//      CHECK: br ^[[EXIT_BLOCK]](%[[ARG0]] : [[ARG0_TYPE]])
//      CHECK: ^[[EXIT_BLOCK]](%{{.*}}: [[ARG0_TYPE]])
// CHECK-NEXT: linalg.copy(%[[ARG1]], %[[RESULT]])
// CHECK-NEXT: return

// -----

// Test Case: Simple case for checking if BufferAssignmentPlacer creates AllocOps right before GenericOps.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @compute_allocs_position_simple
func @compute_allocs_position_simple(%cond: i1, %arg0: tensor<2xf32>) -> tensor<2xf32>{
    %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
      ins(%arg0 : tensor<2xf32>) {
    ^bb0(%gen1_arg0: f32):
      %tmp1 = exp %gen1_arg0 : f32
      linalg.yield %tmp1 : f32
    } -> tensor<2xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
      ins(%0 : tensor<2xf32>) {
    ^bb0(%gen2_arg0: f32):
      %tmp2 = exp %gen2_arg0 : f32
      linalg.yield %tmp2 : f32
    } -> tensor<2xf32>
    return %1 : tensor<2xf32>
}
//      CHECK: (%{{.*}}: {{.*}}, %[[ARG0:.*]]: memref<2xf32>,
// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} ins(%[[ARG0]]{{.*}} outs(%[[FIRST_ALLOC]]
//      CHECK: %[[SECOND_ALLOC:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} ins(%[[FIRST_ALLOC]]{{.*}} outs(%[[SECOND_ALLOC]]

// -----

// Test Case: if-else case for checking if BufferAssignmentPlacer creates AllocOps right before GenericOps.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @compute_allocs_position
func @compute_allocs_position(%cond: i1, %arg0: tensor<2xf32>) -> tensor<2xf32>{
    %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
      ins(%arg0 : tensor<2xf32>) {
    ^bb0(%gen1_arg0: f32):
      %tmp1 = exp %gen1_arg0 : f32
      linalg.yield %tmp1 : f32
    } -> tensor<2xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
      ins(%0 : tensor<2xf32>) {
    ^bb0(%gen2_arg0: f32):
      %tmp2 = exp %gen2_arg0 : f32
      linalg.yield %tmp2 : f32
    } -> tensor<2xf32>
    cond_br %cond, ^bb1(%arg0, %0: tensor<2xf32>, tensor<2xf32>),
                   ^bb2(%0, %arg0: tensor<2xf32>, tensor<2xf32>)
  ^bb1(%arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>):
    %2 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
      ins(%arg0 : tensor<2xf32>) {
    ^bb0(%gen3_arg0: f32):
      %tmp3 = exp %gen3_arg0 : f32
      linalg.yield %tmp3 : f32
    } -> tensor<2xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
      ins(%2 : tensor<2xf32>) {
    ^bb0(%gen4_arg0: f32):
      %tmp4 = exp %gen4_arg0 : f32
      linalg.yield %tmp4 : f32
    } -> tensor<2xf32>
    br ^exit(%arg1, %arg2 : tensor<2xf32>, tensor<2xf32>)
  ^bb2(%arg3 : tensor<2xf32>, %arg4 : tensor<2xf32>):
    %4 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
      ins(%arg0 : tensor<2xf32>) {
    ^bb0(%gen5_arg0: f32):
      %tmp5 = exp %gen5_arg0 : f32
      linalg.yield %tmp5 : f32
    } -> tensor<2xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
      ins(%4 : tensor<2xf32>) {
    ^bb0(%gen6_arg0: f32):
      %tmp6 = exp %gen6_arg0 : f32
      linalg.yield %tmp6 : f32
    } -> tensor<2xf32>
    br ^exit(%arg3, %arg4 : tensor<2xf32>, tensor<2xf32>)
  ^exit(%arg5 : tensor<2xf32>, %arg6 : tensor<2xf32>):
    %6 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
      ins(%arg0 : tensor<2xf32>) {
    ^bb0(%gen7_arg0: f32):
      %tmp7 = exp %gen7_arg0 : f32
      linalg.yield %tmp7 : f32
    } -> tensor<2xf32>
    %7 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
      ins(%6 : tensor<2xf32>) {
    ^bb0(%gen8_arg0: f32):
      %tmp8 = exp %gen8_arg0 : f32
      linalg.yield %tmp8 : f32
    } -> tensor<2xf32>
    return %7 : tensor<2xf32>
}
//      CHECK: (%{{.*}}: {{.*}}, %[[ARG0:.*]]: memref<2xf32>,
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} ins(%[[ARG0]]{{.*}} outs(%[[ALLOC0]]
//      CHECK: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} ins(%[[ALLOC0]]{{.*}} outs(%[[ALLOC1]]
//      CHECK: cond_br %{{.*}}, ^[[BB0:.*]]({{.*}}), ^[[BB1:.*]](
// CHECK-NEXT: ^[[BB0]]
// CHECK-NEXT: %[[ALLOC2:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} ins(%[[ARG0]]{{.*}} outs(%[[ALLOC2]]
//      CHECK: %[[ALLOC3:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} ins(%[[ALLOC2]]{{.*}} outs(%[[ALLOC3]]
//      CHECK: br ^[[EXIT:.*]]({{.*}})
// CHECK-NEXT: ^[[BB1]]
// CHECK-NEXT: %[[ALLOC4:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} ins(%[[ARG0]]{{.*}} outs(%[[ALLOC4]]
//      CHECK: %[[ALLOC5:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} ins(%[[ALLOC4]]{{.*}} outs(%[[ALLOC5]]
//      CHECK: br ^[[EXIT]]
// CHECK-NEXT: ^[[EXIT]]
// CHECK-NEXT: %[[ALLOC6:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} ins(%[[ARG0]]{{.*}} outs(%[[ALLOC6]]
//      CHECK: %[[ALLOC7:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} ins(%[[ALLOC6]]{{.*}} outs(%[[ALLOC7]]

// -----

// Test case: Checking BufferAssignmentCallOpConverter and
// BufferAssignmentFuncOpConverter and BufferAssignmentReturnOpConverter all
// together. The signature of `callee` after signature conversion would be:

// func @callee(%arg0: memref<5xf32>,%arg1: memref<5xf32>) -> ()

// The operands and results of caller and return operations must be matched
// respectively.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @callee
func @callee(%arg1: tensor<5xf32>) -> tensor<5xf32> {
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
    ins(%arg1 : tensor<5xf32>) {
  ^bb0(%gen1_arg0: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  } -> tensor<5xf32>
  return %0 : tensor<5xf32>
}
// CHECK: (%[[CALLEE_ARG:.*]]: memref<5xf32>, %[[CALLEE_RESULT:.*]]: memref<5xf32>)
// CHECK: %[[ALLOC:.*]] = alloc()
// CHECK: linalg.generic
// CHECK: linalg.copy(%[[ALLOC]], %[[CALLEE_RESULT]])
// CHECK: return

// CHECK-LABEL: func @caller
func @caller(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %x = call @callee(%arg0) : (tensor<5xf32>) -> tensor<5xf32>
  %y = call @callee(%x) : (tensor<5xf32>) -> tensor<5xf32>
  return %y : tensor<5xf32>
}
// CHECK: (%[[CALLER_ARG:.*]]: memref<5xf32>, %[[CALLER_RESULT:.*]]: memref<5xf32>)
// CHECK: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK: call @callee(%[[CALLER_ARG]], %[[FIRST_ALLOC]])
// CHECK: %[[SECOND_ALLOC:.*]] = alloc()
// CHECK: call @callee(%[[FIRST_ALLOC]], %[[SECOND_ALLOC]])
// CHECK: linalg.copy(%[[SECOND_ALLOC]], %[[CALLER_RESULT]])
// CHECK: return

// -----

// Test case: Checking BufferAssignmentCallOpConverter and
// BufferAssignmentFuncOpConverter and BufferAssignmentReturnOpConverter all
// together on functions that also have memref typed results. The signature of
// `callee` after signature conversion would be:

// func @callee(%arg0: memref<5xf32>,%arg1: memref<5xf32>)-> memref<2xf32>

// where %arg0 is the input and %arg1 is the output buffer and the original memref
// type result remain as the function result. Then, the rewriter should match the
// caller's signature with the callee. Thus, two buffers will be allocated instead
// of %x0 and %y0 and they are passed to the callers' operands list as the output
// buffers. %x1 and %y1 remain as callers' results.


// CHECK-LABEL: func @callee
func @callee(%arg1: tensor<5xf32>) -> (tensor<5xf32>, memref<2xf32>) {
  %buff = alloc() : memref<2xf32>
  return %arg1, %buff : tensor<5xf32>, memref<2xf32>
}
//      CHECK: (%[[CALLEE_ARG:.*]]: memref<5xf32>, %[[CALLEE_RESULT:.*]]: memref<5xf32>)
// CHECK-SAME: memref<2xf32>
//      CHECK: %[[ALLOC:.*]] = alloc()
//      CHECK: linalg.copy(%[[CALLEE_ARG]], %[[CALLEE_RESULT]])
//      CHECK: return %[[ALLOC]]


// CHECK-LABEL: func @caller
func @caller(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %x0, %x1 = call @callee(%arg0) : (tensor<5xf32>) -> (tensor<5xf32>, memref<2xf32>)
  %y0, %y1 = call @callee(%x0) : (tensor<5xf32>) -> (tensor<5xf32>, memref<2xf32>)
  return %y0 : tensor<5xf32>
}
// CHECK: (%[[CALLER_ARG:.*]]: memref<5xf32>, %[[CALLER_RESULT:.*]]: memref<5xf32>)
// CHECK: %[[X0:.*]] = alloc()
// CHECK: %[[X1:.*]] = call @callee(%[[CALLER_ARG]], %[[X0]])
// CHECK: %[[Y0:.*]] = alloc()
// CHECK: %[[Y1:.*]] = call @callee(%[[X0]], %[[Y0]])
// CHECK: linalg.copy(%[[Y0]], %[[CALLER_RESULT]])
// CHECK: return

// -----

// CHECK-LABEL: func @func_with_unranked_arg
func @func_with_unranked_arg(%arg0: tensor<*xf32>) {
  return
}
// CHECK-SAME: ([[ARG:%.*]]: memref<*xf32>)

// -----

// Test case: Testing BufferAssginmnetCallOpConverter to see if it matches with the
// signature of the new signature of the callee function when there are tuple typed
// args and results. BufferAssginmentTypeConverter is set to flatten tuple typed
// arguments. The tuple typed values should be decomposed and composed using
// get_tuple_element and make_tuple operations of test dialect. Tensor types are
// converted to Memref. Memref typed function results are appended to the function
// arguments list.

// CHECK-LABEL: func @callee
func @callee(%arg0: tuple<tensor<2xf32>,i1, tensor<5xf32>>) -> (tuple<tensor<2xf32>,i1, tensor<5xf32>>){
  return %arg0 : tuple<tensor<2xf32>,i1, tensor<5xf32>>
}
// CHECK-SAME: (%[[ARG0:.*]]: memref<2xf32>, %[[ARG1:.*]]: i1, %[[ARG2:.*]]: memref<5xf32>, %[[RESULT0:.*]]: memref<2xf32>, %[[RESULT1:.*]]: memref<5xf32>)
// CHECK-SAME: i1
// CHECK-NEXT: %[[TUPLE:.*]] = "test.make_tuple"(%[[ARG0]], %[[ARG1]], %[[ARG2]])
// CHECK-NEXT: %[[FIRST_ELEM:.*]] = "test.get_tuple_element"(%[[TUPLE]]) {index = 0 : i32}
// CHECK-NEXT: %[[SECOND_ELEM:.*]] = "test.get_tuple_element"(%[[TUPLE]]) {index = 1 : i32}
// CHECK-NEXT: %[[THIRD_ELEM:.*]]  = "test.get_tuple_element"(%[[TUPLE]]) {index = 2 : i32}
// CHECK-NEXT: linalg.copy(%[[FIRST_ELEM]], %[[RESULT0]])
// CHECK-NEXT: linalg.copy(%[[THIRD_ELEM]], %[[RESULT1]])
// CHECK-NEXT: return %[[SECOND_ELEM]]


// CHECK-LABEL: func @caller
func @caller(%arg0: tuple<tensor<2xf32>,i1, tensor<5xf32>>) -> tuple<tensor<2xf32>,i1, tensor<5xf32>>{
  %x0 = call @callee(%arg0) : (tuple<tensor<2xf32>,i1, tensor<5xf32>>) -> (tuple<tensor<2xf32>,i1, tensor<5xf32>>)
  %y0 = call @callee(%x0) : (tuple<tensor<2xf32>,i1, tensor<5xf32>>) -> (tuple<tensor<2xf32>,i1, tensor<5xf32>>)
  return %y0 : tuple<tensor<2xf32>,i1, tensor<5xf32>>
}
// CHECK-SAME: (%[[ARG0:.*]]: memref<2xf32>, %[[ARG1:.*]]: i1, %[[ARG2:.*]]: memref<5xf32>, %[[RESULT0:.*]]: memref<2xf32>, %[[RESULT1:.*]]: memref<5xf32>)
// CHECK-SAME: i1
// CHECK-NEXT: %[[TUPLE:.*]] = "test.make_tuple"(%[[ARG0]], %[[ARG1]], %[[ARG2]])
// CHECK-NEXT: %[[FIRST_ELEM:.*]] = "test.get_tuple_element"(%[[TUPLE]]) {index = 0 : i32}
// CHECK-NEXT: %[[SECOND_ELEM:.*]] = "test.get_tuple_element"(%[[TUPLE]]) {index = 1 : i32}
// CHECK-NEXT: %[[THIRD_ELEM:.*]]  = "test.get_tuple_element"(%[[TUPLE]]) {index = 2 : i32}
// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK-NEXT: %[[SECOND_ALLOC:.*]] = alloc()
// CHECK-NEXT: %[[CALLEE_RESULT:.*]] = call @callee(%[[FIRST_ELEM]], %[[SECOND_ELEM]], %[[THIRD_ELEM]], %[[FIRST_ALLOC]], %[[SECOND_ALLOC]])
// CHECK-SAME: (memref<2xf32>, i1, memref<5xf32>, memref<2xf32>, memref<5xf32>) -> i1
// CHECK-NEXT: %[[TUPLE:.*]] = "test.make_tuple"(%[[FIRST_ALLOC]], %[[CALLEE_RESULT]], %[[SECOND_ALLOC]])
// CHECK-NEXT: %[[FIRST_ELEM:.*]] = "test.get_tuple_element"(%[[TUPLE]]) {index = 0 : i32}
// CHECK-NEXT: %[[SECOND_ELEM:.*]] = "test.get_tuple_element"(%[[TUPLE]]) {index = 1 : i32}
// CHECK-NEXT: %[[THIRD_ELEM:.*]]  = "test.get_tuple_element"(%[[TUPLE]]) {index = 2 : i32}
// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK-NEXT: %[[SECOND_ALLOC:.*]] = alloc()
// CHECK-NEXT: %[[CALLEE_RESULT:.*]] = call @callee(%[[FIRST_ELEM]], %[[SECOND_ELEM]], %[[THIRD_ELEM]], %[[FIRST_ALLOC]], %[[SECOND_ALLOC]])
// CHECK-SAME: (memref<2xf32>, i1, memref<5xf32>, memref<2xf32>, memref<5xf32>) -> i1
// CHECK-NEXT: %[[TUPLE:.*]] = "test.make_tuple"(%[[FIRST_ALLOC]], %[[CALLEE_RESULT]], %[[SECOND_ALLOC]])
// CHECK-NEXT: %[[FIRST_ELEM:.*]] = "test.get_tuple_element"(%[[TUPLE]]) {index = 0 : i32}
// CHECK-NEXT: %[[SECOND_ELEM:.*]] = "test.get_tuple_element"(%[[TUPLE]]) {index = 1 : i32}
// CHECK-NEXT: %[[THIRD_ELEM:.*]]  = "test.get_tuple_element"(%[[TUPLE]]) {index = 2 : i32}
// CHECK-NEXT: linalg.copy(%[[FIRST_ELEM]], %[[RESULT0]])
// CHECK-NEXT: linalg.copy(%[[THIRD_ELEM]], %[[RESULT1]])
// CHECK-NEXT: return %[[SECOND_ELEM]]

// -----

// Test case: Testing BufferAssginmnetFuncOpConverter and
// BufferAssginmentReturnOpConverter to see if the return operation matches with
// the new function signature when there are tuple typed args and results.
// BufferAssginmentTypeConverter is set to flatten tuple typed arguments. The tuple
// typed values should be decomposed and composed using get_tuple_element and
// make_tuple operations of test dialect. Tensor types are converted to Memref.
// Memref typed function results are appended to the function arguments list.

// CHECK-LABEL: func @decompose_tuple_typed_function_args_and_results
func @decompose_tuple_typed_function_args_and_results(%arg0: tuple<i1,f32>, %arg1: tensor<10xf32>, %arg2: tuple<i1, tensor<5xf32>>) -> (tuple<i1, tensor<5xf32>>, tensor<10xf32>, tuple<i1,f32>){
  return %arg2, %arg1, %arg0 : tuple<i1, tensor<5xf32>>, tensor<10xf32>, tuple<i1,f32>
}
// CHECK-SAME: %[[ARG0:.*]]: i1, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: memref<10xf32>, %[[ARG3:.*]]: i1, %[[ARG4:.*]]: memref<5xf32>, %[[RESULT0:.*]]: memref<5xf32>, %[[RESULT1:.*]]: memref<10xf32>
// CHECK-SAME: (i1, i1, f32)
// CHECK-NEXT: %[[FIRST_TUPLE:.*]] = "test.make_tuple"(%[[ARG0]], %[[ARG1]])
// CHECK-NEXT: %[[SECOND_TUPLE:.*]] = "test.make_tuple"(%[[ARG3]], %[[ARG4]])
// CHECK-NEXT: %[[SECOND_TUPLE_FIRST_ELEM:.*]]  = "test.get_tuple_element"(%[[SECOND_TUPLE]]) {index = 0 : i32}
// CHECK-NEXT: %[[SECOND_TUPLE_SECOND_ELEM:.*]] = "test.get_tuple_element"(%[[SECOND_TUPLE]]) {index = 1 : i32}
// CHECK-NEXT: %[[FIRST_TUPLE_FIRST_ELEM:.*]] = "test.get_tuple_element"(%[[FIRST_TUPLE]]) {index = 0 : i32}
// CHECK-NEXT: %[[FIRST_TUPLE_SECOND_ELEM:.*]] = "test.get_tuple_element"(%[[FIRST_TUPLE]]) {index = 1 : i32}
// CHECK-NEXT: linalg.copy(%[[SECOND_TUPLE_SECOND_ELEM]], %[[RESULT0]])
// CHECK-NEXT: linalg.copy(%[[ARG2]], %[[RESULT1]])
// CHECK-NEXT: return %[[SECOND_TUPLE_FIRST_ELEM]], %[[FIRST_TUPLE_FIRST_ELEM]], %[[FIRST_TUPLE_SECOND_ELEM]]
