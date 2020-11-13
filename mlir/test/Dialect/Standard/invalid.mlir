// RUN: mlir-opt -split-input-file %s -verify-diagnostics

func @test_index_cast_shape_error(%arg0 : tensor<index>) -> tensor<2xi64> {
  // expected-error @+1 {{all non-scalar operands/results must have the same shape and base type: found 'tensor<2xi64>' and 'tensor<index>'}}
  %0 = index_cast %arg0 : tensor<index> to tensor<2xi64>
  return %0 : tensor<2xi64>
}

// -----

func @test_index_cast_tensor_error(%arg0 : tensor<index>) -> i64 {
  // expected-error @+1 {{if an operand is non-scalar, then there must be at least one non-scalar result}}
  %0 = index_cast %arg0 : tensor<index> to i64
  return %0 : i64
}

// -----

func @dynamic_tensor_from_elements(%m : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{must have as many index operands as dynamic extents in the result type}}
  %tnsr = dynamic_tensor_from_elements %m {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = constant 8.0 : f32
      yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func @dynamic_tensor_from_elements(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{must have one body argument per input dimension}}
  %tnsr = dynamic_tensor_from_elements %m, %n {
    ^bb0(%i : index, %j : index):
      %elem = constant 8.0 : f32
      yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func @dynamic_tensor_from_elements(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{all body arguments must be index}}
  %tnsr = dynamic_tensor_from_elements %m, %n {
    ^bb0(%i : index, %j : index, %k : i64):
      %elem = constant 8.0 : f32
      yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func @dynamic_tensor_from_elements(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+2 {{op expects regions to end with 'std.yield', found 'std.return'}}
  // expected-note @+1 {{in custom textual format, the absence of terminator implies 'std.yield'}}
  %tnsr = dynamic_tensor_from_elements %m, %n {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = constant 8.0 : f32
      return %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func @dynamic_tensor_from_elements(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{body must be terminated with a `yield` operation of the tensor element type}}
  %tnsr = dynamic_tensor_from_elements %m, %n {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = constant 8 : i32
      yield %elem : i32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func @transpose_not_permutation(%v : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>) {
  // expected-error @+1 {{expected a permutation map}}
  transpose %v (i, j) -> (i, i) : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>> to memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>
}

// -----

func @transpose_bad_rank(%v : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>) {
  // expected-error @+1 {{expected a permutation map of same rank as the input}}
  transpose %v (i) -> (i) : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>> to memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>
}

// -----

func @transpose_wrong_type(%v : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>) {
  // expected-error @+1 {{output type 'memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>' does not match transposed input type 'memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>'}}
  transpose %v (i, j) -> (j, i) : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>> to memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>
}

// -----

func @memref_reinterpret_cast_too_many_offsets(%in: memref<?xf32>) {
  // expected-error @+1 {{expected 1 offset values}}
  %out = memref_reinterpret_cast %in to
           offset: [0, 0], sizes: [10, 10], strides: [10, 1]
           : memref<?xf32> to memref<10x10xf32, offset: 0, strides: [10, 1]>
  return
}

// -----

func @memref_reinterpret_cast_incompatible_element_types(%in: memref<*xf32>) {
  // expected-error @+1 {{different element types specified}}
  %out = memref_reinterpret_cast %in to
           offset: [0], sizes: [10], strides: [1]
         : memref<*xf32> to memref<10xi32, offset: 0, strides: [1]>
  return
}

// -----

func @memref_reinterpret_cast_incompatible_memory_space(%in: memref<*xf32>) {
  // expected-error @+1 {{different memory spaces specified}}
  %out = memref_reinterpret_cast %in to
           offset: [0], sizes: [10], strides: [1]
         : memref<*xf32> to memref<10xi32, offset: 0, strides: [1], 2>
  return
}

// -----

func @memref_reinterpret_cast_offset_mismatch(%in: memref<?xf32>) {
  // expected-error @+1 {{expected result type with offset = 2 instead of 1}}
  %out = memref_reinterpret_cast %in to
           offset: [1], sizes: [10], strides: [1]
         : memref<?xf32> to memref<10xf32, offset: 2, strides: [1]>
  return
}

// -----

func @memref_reinterpret_cast_size_mismatch(%in: memref<*xf32>) {
  // expected-error @+1 {{expected result type with size = 10 instead of 1 in dim = 0}}
  %out = memref_reinterpret_cast %in to
           offset: [0], sizes: [10], strides: [1]
         : memref<*xf32> to memref<1xf32, offset: 0, strides: [1]>
  return
}

// -----

func @memref_reinterpret_cast_offset_mismatch(%in: memref<?xf32>) {
  // expected-error @+1 {{expected result type with stride = 2 instead of 1 in dim = 0}}
  %out = memref_reinterpret_cast %in to
           offset: [2], sizes: [10], strides: [2]
         : memref<?xf32> to memref<10xf32, offset: 2, strides: [1]>
  return
}

// -----

func @memref_reinterpret_cast_offset_mismatch(%in: memref<?xf32>) {
  %c0 = constant 0 : index
  %c10 = constant 10 : index
  // expected-error @+1 {{expected result type with size = 10 instead of -1 in dim = 0}}
  %out = memref_reinterpret_cast %in to
           offset: [%c0], sizes: [10, %c10], strides: [%c10, 1]
           : memref<?xf32> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}

// -----

func @memref_reshape_element_type_mismatch(
       %buf: memref<*xf32>, %shape: memref<1xi32>) {
  // expected-error @+1 {{element types of source and destination memref types should be the same}}
  memref_reshape %buf(%shape) : (memref<*xf32>, memref<1xi32>) -> memref<?xi32>
}

// -----

func @memref_reshape_dst_ranked_shape_unranked(
       %buf: memref<*xf32>, %shape: memref<?xi32>) {
  // expected-error @+1 {{cannot use shape operand with dynamic length to reshape to statically-ranked memref type}}
  memref_reshape %buf(%shape) : (memref<*xf32>, memref<?xi32>) -> memref<?xf32>
}

// -----

func @memref_reshape_dst_shape_rank_mismatch(
       %buf: memref<*xf32>, %shape: memref<1xi32>) {
  // expected-error @+1 {{length of shape operand differs from the result's memref rank}}
  memref_reshape %buf(%shape)
    : (memref<*xf32>, memref<1xi32>) -> memref<?x?xf32>
}

// -----

func @memref_reshape_src_affine_map_is_not_identity(
        %buf: memref<4x4xf32, offset: 0, strides: [3, 2]>,
        %shape: memref<1xi32>) {
  // expected-error @+1 {{source memref type should have identity affine map}}
  memref_reshape %buf(%shape)
    : (memref<4x4xf32, offset: 0, strides: [3, 2]>, memref<1xi32>)
    -> memref<8xf32>
}

// -----

func @memref_reshape_result_affine_map_is_not_identity(
        %buf: memref<4x4xf32>, %shape: memref<1xi32>) {
  // expected-error @+1 {{result memref type should have identity affine map}}
  memref_reshape %buf(%shape)
    : (memref<4x4xf32>, memref<1xi32>) -> memref<8xf32, offset: 0, strides: [2]>
}

// -----

// expected-error @+1 {{type should be static shaped memref}}
global_memref @foo : i32

// -----

// expected-error @+1 {{type should be static shaped memref}}
global_memref @foo : i32 = 5

// -----

// expected-error @+1 {{type should be static shaped memref}}
global_memref @foo : memref<*xf32>

// -----

// expected-error @+1 {{type should be static shaped memref}}
global_memref @foo : memref<?x?xf32>

// -----

// expected-error @+1 {{initial value should be a unit or elements attribute}}
global_memref @foo : memref<2x2xf32>  = "foo"

// -----

// expected-error @+1 {{inferred shape of elements literal ([2]) does not match type ([2, 2])}}
global_memref @foo : memref<2x2xf32> = dense<[0.0, 1.0]>

// -----

// expected-error @+1 {{expected valid '@'-identifier for symbol name}}
global_memref "private" "public" @foo : memref<2x2xf32>  = "foo"

// -----

// expected-error @+1 {{expected valid '@'-identifier for symbol name}}
global_memref constant external @foo : memref<2x2xf32>  = "foo"

// -----

// constant qualifier must be after visibility.
// expected-error @+1 {{expected valid '@'-identifier for symbol name}}
global_memref constant "private" @foo : memref<2x2xf32>  = "foo"


// -----

// expected-error @+1 {{op visibility expected to be one of ["public", "private", "nested"], but got "priate"}}
global_memref "priate" constant @memref5 : memref<2xf32>  = uninitialized

// -----

func @nonexistent_global_memref() {
  // expected-error @+1 {{'gv' does not reference a valid global memref}}
  %0 = get_global_memref @gv : memref<3xf32>
  return
}

// -----

func @foo()

func @nonexistent_global_memref() {
  // expected-error @+1 {{'foo' does not reference a valid global memref}}
  %0 = get_global_memref @foo : memref<3xf32>
  return
}

// -----

global_memref @gv : memref<3xi32>

func @mismatched_types() {
  // expected-error @+1 {{result type 'memref<3xf32>' does not match type 'memref<3xi32>' of the global memref @gv}}
  %0 = get_global_memref @gv : memref<3xf32>
  return
}

