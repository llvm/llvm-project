// RUN: mlir-opt -split-input-file %s -verify-diagnostics

func @transpose_not_permutation(%v : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>) {
  // expected-error @+1 {{expected a permutation map}}
  memref.transpose %v (i, j) -> (i, i) : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>> to memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>
}

// -----

func @transpose_bad_rank(%v : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>) {
  // expected-error @+1 {{expected a permutation map of same rank as the input}}
  memref.transpose %v (i) -> (i) : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>> to memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>
}

// -----

func @transpose_wrong_type(%v : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>) {
  // expected-error @+1 {{output type 'memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>' does not match transposed input type 'memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>'}}
  memref.transpose %v (i, j) -> (j, i) : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>> to memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>
}

// -----

func @memref_reinterpret_cast_too_many_offsets(%in: memref<?xf32>) {
  // expected-error @+1 {{expected <= 1 offset values}}
  %out = memref.reinterpret_cast %in to
           offset: [0, 0], sizes: [10, 10], strides: [10, 1]
           : memref<?xf32> to memref<10x10xf32, offset: 0, strides: [10, 1]>
  return
}

// -----

func @memref_reinterpret_cast_incompatible_element_types(%in: memref<*xf32>) {
  // expected-error @+1 {{different element types specified}}
  %out = memref.reinterpret_cast %in to
           offset: [0], sizes: [10], strides: [1]
         : memref<*xf32> to memref<10xi32, offset: 0, strides: [1]>
  return
}

// -----

func @memref_reinterpret_cast_incompatible_memory_space(%in: memref<*xf32>) {
  // expected-error @+1 {{different memory spaces specified}}
  %out = memref.reinterpret_cast %in to
           offset: [0], sizes: [10], strides: [1]
         : memref<*xf32> to memref<10xi32, offset: 0, strides: [1], 2>
  return
}

// -----

func @memref_reinterpret_cast_offset_mismatch(%in: memref<?xf32>) {
  // expected-error @+1 {{expected result type with offset = 2 instead of 1}}
  %out = memref.reinterpret_cast %in to
           offset: [1], sizes: [10], strides: [1]
         : memref<?xf32> to memref<10xf32, offset: 2, strides: [1]>
  return
}

// -----

func @memref_reinterpret_cast_size_mismatch(%in: memref<*xf32>) {
  // expected-error @+1 {{expected result type with size = 10 instead of 1 in dim = 0}}
  %out = memref.reinterpret_cast %in to
           offset: [0], sizes: [10], strides: [1]
         : memref<*xf32> to memref<1xf32, offset: 0, strides: [1]>
  return
}

// -----

func @memref_reinterpret_cast_offset_mismatch(%in: memref<?xf32>) {
  // expected-error @+1 {{expected result type with stride = 2 instead of 1 in dim = 0}}
  %out = memref.reinterpret_cast %in to
           offset: [2], sizes: [10], strides: [2]
         : memref<?xf32> to memref<10xf32, offset: 2, strides: [1]>
  return
}

// -----

func @memref_reinterpret_cast_offset_mismatch(%in: memref<?xf32>) {
  %c0 = constant 0 : index
  %c10 = constant 10 : index
  // expected-error @+1 {{expected result type with size = 10 instead of -1 in dim = 0}}
  %out = memref.reinterpret_cast %in to
           offset: [%c0], sizes: [10, %c10], strides: [%c10, 1]
           : memref<?xf32> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}

// -----

func @memref_reshape_element_type_mismatch(
       %buf: memref<*xf32>, %shape: memref<1xi32>) {
  // expected-error @+1 {{element types of source and destination memref types should be the same}}
  memref.reshape %buf(%shape) : (memref<*xf32>, memref<1xi32>) -> memref<?xi32>
}

// -----

func @memref_reshape_dst_ranked_shape_unranked(
       %buf: memref<*xf32>, %shape: memref<?xi32>) {
  // expected-error @+1 {{cannot use shape operand with dynamic length to reshape to statically-ranked memref type}}
  memref.reshape %buf(%shape) : (memref<*xf32>, memref<?xi32>) -> memref<?xf32>
}

// -----

func @memref_reshape_dst_shape_rank_mismatch(
       %buf: memref<*xf32>, %shape: memref<1xi32>) {
  // expected-error @+1 {{length of shape operand differs from the result's memref rank}}
  memref.reshape %buf(%shape)
    : (memref<*xf32>, memref<1xi32>) -> memref<?x?xf32>
}

// -----

func @memref_reshape_src_affine_map_is_not_identity(
        %buf: memref<4x4xf32, offset: 0, strides: [3, 2]>,
        %shape: memref<1xi32>) {
  // expected-error @+1 {{source memref type should have identity affine map}}
  memref.reshape %buf(%shape)
    : (memref<4x4xf32, offset: 0, strides: [3, 2]>, memref<1xi32>)
    -> memref<8xf32>
}

// -----

func @memref_reshape_result_affine_map_is_not_identity(
        %buf: memref<4x4xf32>, %shape: memref<1xi32>) {
  // expected-error @+1 {{result memref type should have identity affine map}}
  memref.reshape %buf(%shape)
    : (memref<4x4xf32>, memref<1xi32>) -> memref<8xf32, offset: 0, strides: [2]>
}

// -----

// expected-error @+1 {{type should be static shaped memref}}
memref.global @foo : i32

// -----

// expected-error @+1 {{type should be static shaped memref}}
memref.global @foo : i32 = 5

// -----

// expected-error @+1 {{type should be static shaped memref}}
memref.global @foo : memref<*xf32>

// -----

// expected-error @+1 {{type should be static shaped memref}}
memref.global @foo : memref<?x?xf32>

// -----

// expected-error @+1 {{initial value should be a unit or elements attribute}}
memref.global @foo : memref<2x2xf32>  = "foo"

// -----

// expected-error @+1 {{inferred shape of elements literal ([2]) does not match type ([2, 2])}}
memref.global @foo : memref<2x2xf32> = dense<[0.0, 1.0]>

// -----

// expected-error @+1 {{expected valid '@'-identifier for symbol name}}
memref.global "private" "public" @foo : memref<2x2xf32>  = "foo"

// -----

// expected-error @+1 {{expected valid '@'-identifier for symbol name}}
memref.global constant external @foo : memref<2x2xf32>  = "foo"

// -----

// constant qualifier must be after visibility.
// expected-error @+1 {{expected valid '@'-identifier for symbol name}}
memref.global constant "private" @foo : memref<2x2xf32>  = "foo"


// -----

// expected-error @+1 {{op visibility expected to be one of ["public", "private", "nested"], but got "priate"}}
memref.global "priate" constant @memref5 : memref<2xf32>  = uninitialized

// -----

func @nonexistent_global_memref() {
  // expected-error @+1 {{'gv' does not reference a valid global memref}}
  %0 = memref.get_global @gv : memref<3xf32>
  return
}

// -----

func @foo()

func @nonexistent_global_memref() {
  // expected-error @+1 {{'foo' does not reference a valid global memref}}
  %0 = memref.get_global @foo : memref<3xf32>
  return
}

// -----

memref.global @gv : memref<3xi32>

func @mismatched_types() {
  // expected-error @+1 {{result type 'memref<3xf32>' does not match type 'memref<3xi32>' of the global memref @gv}}
  %0 = memref.get_global @gv : memref<3xf32>
  return
}

// -----

func @copy_different_shape(%arg0: memref<2xf32>, %arg1: memref<3xf32>) {
  // expected-error @+1 {{op requires the same shape for all operands}}
  memref.copy %arg0, %arg1 : memref<2xf32> to memref<3xf32>
  return
}

// -----

func @copy_different_eltype(%arg0: memref<2xf32>, %arg1: memref<2xf16>) {
  // expected-error @+1 {{op requires the same element type for all operands}}
  memref.copy %arg0, %arg1 : memref<2xf32> to memref<2xf16>
  return
}

// -----

func @expand_shape(%arg0: memref<f32>) {
  // expected-error @+1 {{expected non-zero memref ranks}}
  %0 = memref.expand_shape %arg0 [[0]] : memref<f32> into memref<f32>
}

// -----

func @collapse_shape_to_higher_rank(%arg0: memref<f32>) {
  // expected-error @+1 {{expected the type 'memref<f32>' to have higher rank than the type = 'memref<1xf32>'}}
  %0 = memref.collapse_shape %arg0 [[0]] : memref<f32> into memref<1xf32>
}

// -----

func @expand_shape_to_smaller_rank(%arg0: memref<1xf32>) {
  // expected-error @+1 {{expected the type 'memref<f32>' to have higher rank than the type = 'memref<1xf32>'}}
  %0 = memref.expand_shape %arg0 [[0]] : memref<1xf32> into memref<f32>
}

// -----

func @collapse_shape(%arg0: memref<?xf32>) {
  // expected-error @+1 {{expected to collapse or expand dims}}
  %0 = memref.collapse_shape %arg0 [[0]] : memref<?xf32> into memref<?xf32>
}

// -----

func @collapse_shape_mismatch_indices_num(%arg0: memref<?x?x?xf32>) {
  // expected-error @+1 {{expected rank of the collapsed type(2) to be the number of reassociation maps(1)}}
  %0 = memref.collapse_shape %arg0 [[0, 1]] :
    memref<?x?x?xf32> into memref<?x?xf32, offset: 0, strides: [?, 1]>
}

// -----

func @collapse_shape_invalid_reassociation(%arg0: memref<?x?x?xf32>) {
  // expected-error @+1 {{expected reassociation map #1 to be valid and contiguous}}
  %0 = memref.collapse_shape %arg0 [[0, 1], [1, 2]] :
    memref<?x?x?xf32> into memref<?x?xf32, offset: 0, strides: [?, 1]>
}

// -----

func @collapse_shape_wrong_collapsed_type(%arg0: memref<?x?x?xf32>) {
  // expected-error @+1 {{expected collapsed type to be 'memref<?x?xf32>', but got 'memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>>'}}
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] :
    memref<?x?x?xf32> into memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>>
}

// -----

func @expand_shape_illegal_dynamic_memref
  (%arg0: memref<?x?x?xf32>) -> memref<?x?x?x4x?xf32> {
  // expected-error @+1 {{invalid to have a single dimension (2) expanded into multiple dynamic dims (2,4)}}
  %0 = memref.expand_shape %arg0 [[0], [1], [2, 3, 4]]
      : memref<?x?x?xf32> into memref<?x?x?x4x?xf32>
  return %0 : memref<?x?x?x4x?xf32>
}

// -----

func @expand_shape_illegal_static_memref
  (%arg0: memref<2x3x20xf32>) -> memref<2x3x2x4x5xf32> {
  // expected-error @+1 {{expected dimension 2 of collapsed type to be static value of 40}}
  %0 = memref.expand_shape %arg0 [[0], [1], [2, 3, 4]]
      : memref<2x3x20xf32> into memref<2x3x2x4x5xf32>
  return %0 : memref<2x3x2x4x5xf32>
}

// -----

func @collapse_shape_illegal_static_memref
  (%arg0: memref<2x3x2x4x5xf32>) -> memref<2x3x20xf32> {
  // expected-error @+1 {{expected dimension 2 of collapsed type to be static value of 40}}
  %0 = memref.collapse_shape %arg0 [[0], [1], [2, 3, 4]]
      : memref<2x3x2x4x5xf32> into memref<2x3x20xf32>
  return %0 : memref<2x3x20xf32>
}

// -----

func @expand_shape_illegal_mixed_memref(%arg0 : memref<?x?xf32>)
    -> memref<?x4x5xf32> {
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 5}}
  %0 = memref.expand_shape %arg0 [[0, 1], [2]]
      : memref<?x?xf32> into memref<?x4x5xf32>
  return %0 : memref<?x4x5xf32>
}

// -----

func @expand_shape_illegal_mixed_memref_2(%arg0 : memref<?x?xf32>)
    -> memref<?x4x5xf32> {
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 20}}
  %0 = memref.expand_shape %arg0 [[0], [1, 2]]
      : memref<?x?xf32> into memref<?x4x5xf32>
  return %0 : memref<?x4x5xf32>
}

// -----

func @collapse_shape_illegal_mixed_memref(%arg0 : memref<?x4x5xf32>)
    -> memref<?x?xf32> {
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 5}}
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]]
      : memref<?x4x5xf32> into memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// -----

func @collapse_shape_illegal_mixed_memref_2(%arg0 : memref<?x4x5xf32>)
    -> memref<?x?xf32> {
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 20}}
  %0 = memref.collapse_shape %arg0 [[0], [1, 2]]
      : memref<?x4x5xf32> into memref<?x?xf32>
  return %0 : memref<?x?xf32>
}
