// RUN: mlir-opt -allow-unregistered-dialect -split-input-file %s -verify-diagnostics

func.func @dma_start_not_enough_operands() {
  // expected-error@+1 {{expected at least 4 operands}}
  "memref.dma_start"() : () -> ()
}

// -----

func.func @dma_no_src_memref(%m : f32, %tag : f32, %c0 : index) {
  // expected-error@+1 {{expected source to be of memref type}}
  memref.dma_start %m[%c0], %m[%c0], %c0, %tag[%c0] : f32, f32, f32
}

// -----

func.func @dma_start_not_enough_operands_for_src(
    %src: memref<2x2x2xf32>, %idx: index) {
  // expected-error@+1 {{expected at least 7 operands}}
  "memref.dma_start"(%src, %idx, %idx, %idx) : (memref<2x2x2xf32>, index, index, index) -> ()
}

// -----

func.func @dma_start_src_index_wrong_type(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<i32,2>, %flt: f32) {
  // expected-error@+1 {{expected source indices to be of index type}}
  "memref.dma_start"(%src, %idx, %flt, %dst, %idx, %tag, %idx)
      : (memref<2x2xf32>, index, f32, memref<2xf32,1>, index, memref<i32,2>, index) -> ()
}

// -----

func.func @dma_no_dst_memref(%m : f32, %tag : f32, %c0 : index) {
  %mref = memref.alloc() : memref<8 x f32>
  // expected-error@+1 {{expected destination to be of memref type}}
  memref.dma_start %mref[%c0], %m[%c0], %c0, %tag[%c0] : memref<8 x f32>, f32, f32
}

// -----

func.func @dma_start_not_enough_operands_for_dst(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<i32,2>) {
  // expected-error@+1 {{expected at least 7 operands}}
  "memref.dma_start"(%src, %idx, %idx, %dst, %idx, %idx)
      : (memref<2x2xf32>, index, index, memref<2xf32,1>, index, index) -> ()
}

// -----

func.func @dma_start_dst_index_wrong_type(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<i32,2>, %flt: f32) {
  // expected-error@+1 {{expected destination indices to be of index type}}
  "memref.dma_start"(%src, %idx, %idx, %dst, %flt, %tag, %idx)
      : (memref<2x2xf32>, index, index, memref<2xf32,1>, f32, memref<i32,2>, index) -> ()
}

// -----

func.func @dma_start_dst_index_wrong_type(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<i32,2>, %flt: f32) {
  // expected-error@+1 {{expected num elements to be of index type}}
  "memref.dma_start"(%src, %idx, %idx, %dst, %idx, %flt, %tag)
      : (memref<2x2xf32>, index, index, memref<2xf32,1>, index, f32, memref<i32,2>) -> ()
}

// -----

func.func @dma_no_tag_memref(%tag : f32, %c0 : index) {
  %mref = memref.alloc() : memref<8 x f32>
  // expected-error@+1 {{expected tag to be of memref type}}
  memref.dma_start %mref[%c0], %mref[%c0], %c0, %tag[%c0] : memref<8 x f32>, memref<8 x f32>, f32
}

// -----

func.func @dma_start_not_enough_operands_for_tag(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<2xi32,2>) {
  // expected-error@+1 {{expected at least 8 operands}}
  "memref.dma_start"(%src, %idx, %idx, %dst, %idx, %idx, %tag)
      : (memref<2x2xf32>, index, index, memref<2xf32,1>, index, index, memref<2xi32,2>) -> ()
}

// -----

func.func @dma_start_dst_index_wrong_type(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<2xi32,2>, %flt: f32) {
  // expected-error@+1 {{expected tag indices to be of index type}}
  "memref.dma_start"(%src, %idx, %idx, %dst, %idx, %idx, %tag, %flt)
      : (memref<2x2xf32>, index, index, memref<2xf32,1>, index, index, memref<2xi32,2>, f32) -> ()
}

// -----

func.func @dma_start_too_many_operands(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<i32,2>) {
  // expected-error@+1 {{incorrect number of operands}}
  "memref.dma_start"(%src, %idx, %idx, %dst, %idx, %idx, %tag, %idx, %idx, %idx)
      : (memref<2x2xf32>, index, index, memref<2xf32,1>, index, index, memref<i32,2>, index, index, index) -> ()
}


// -----

func.func @dma_start_wrong_stride_type(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<i32,2>, %flt: f32) {
  // expected-error@+1 {{expected stride and num elements per stride to be of type index}}
  "memref.dma_start"(%src, %idx, %idx, %dst, %idx, %idx, %tag, %idx, %flt)
      : (memref<2x2xf32>, index, index, memref<2xf32,1>, index, index, memref<i32,2>, index, f32) -> ()
}

// -----

func.func @dma_wait_wrong_index_type(%tag : memref<2x2xi32>, %idx: index, %flt: index) {
  // expected-error@+1 {{expected tagIndices to have the same number of elements as the tagMemRef rank, expected 2, but got 1}}
  "memref.dma_wait"(%tag, %flt, %idx) : (memref<2x2xi32>, index, index) -> ()
  return
}

// -----

func.func @transpose_not_permutation(%v : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>) {
  // expected-error @+1 {{expected a permutation map}}
  memref.transpose %v (i, j) -> (i, i) : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>> to memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>
}

// -----

func.func @transpose_bad_rank(%v : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>) {
  // expected-error @+1 {{expected a permutation map of same rank as the input}}
  memref.transpose %v (i) -> (i) : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>> to memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>
}

// -----

func.func @transpose_wrong_type(%v : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>) {
  // expected-error @+1 {{result type 'memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>' is not equivalent to the canonical transposed input type 'memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 + s0 + d1 * s1)>>'}}
  memref.transpose %v (i, j) -> (j, i) : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>> to memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>
}

// -----

func.func @memref_reinterpret_cast_too_many_offsets(%in: memref<?xf32>) {
  // expected-error @+1 {{expected 1 offset values}}
  %out = memref.reinterpret_cast %in to
           offset: [0, 0], sizes: [10, 10], strides: [10, 1]
           : memref<?xf32> to memref<10x10xf32, strided<[10, 1], offset: 0>>
  return
}

// -----

func.func @memref_reinterpret_cast_incompatible_element_types(%in: memref<*xf32>) {
  // expected-error @+1 {{different element types specified}}
  %out = memref.reinterpret_cast %in to
           offset: [0], sizes: [10], strides: [1]
         : memref<*xf32> to memref<10xi32, strided<[1], offset: 0>>
  return
}

// -----

func.func @memref_reinterpret_cast_incompatible_memory_space(%in: memref<*xf32>) {
  // expected-error @+1 {{different memory spaces specified}}
  %out = memref.reinterpret_cast %in to
           offset: [0], sizes: [10], strides: [1]
         : memref<*xf32> to memref<10xi32, strided<[1], offset: 0>, 2>
  return
}

// -----

func.func @memref_reinterpret_cast_offset_mismatch(%in: memref<?xf32>) {
  // expected-error @+1 {{expected result type with offset = 1 instead of 2}}
  %out = memref.reinterpret_cast %in to
           offset: [1], sizes: [10], strides: [1]
         : memref<?xf32> to memref<10xf32, strided<[1], offset: 2>>
  return
}

// -----

func.func @memref_reinterpret_cast_size_mismatch(%in: memref<*xf32>) {
  // expected-error @+1 {{expected result type with size = 10 instead of 1 in dim = 0}}
  %out = memref.reinterpret_cast %in to
           offset: [0], sizes: [10], strides: [1]
         : memref<*xf32> to memref<1xf32, strided<[1], offset: 0>>
  return
}

// -----

func.func @memref_reinterpret_cast_offset_mismatch(%in: memref<?xf32>) {
  // expected-error @+1 {{expected result type with stride = 2 instead of 1 in dim = 0}}
  %out = memref.reinterpret_cast %in to
           offset: [2], sizes: [10], strides: [2]
         : memref<?xf32> to memref<10xf32, strided<[1], offset: 2>>
  return
}

// -----

func.func @memref_reinterpret_cast_no_map_but_offset(%in: memref<?xf32>) {
  // expected-error @+1 {{expected result type with offset = 2 instead of 0}}
  %out = memref.reinterpret_cast %in to offset: [2], sizes: [10], strides: [1]
         : memref<?xf32> to memref<10xf32>
  return
}

// -----

func.func @memref_reinterpret_cast_no_map_but_stride(%in: memref<?xf32>) {
  // expected-error @+1 {{expected result type with stride = 10 instead of 1 in dim = 0}}
  %out = memref.reinterpret_cast %in to offset: [0], sizes: [10], strides: [10]
         : memref<?xf32> to memref<10xf32>
  return
}

// -----

func.func @memref_reinterpret_cast_no_map_but_strides(%in: memref<?x?xf32>) {
  // expected-error @+1 {{expected result type with stride = 42 instead of 10 in dim = 0}}
  %out = memref.reinterpret_cast %in to
           offset: [0], sizes: [9, 10], strides: [42, 1]
         : memref<?x?xf32> to memref<9x10xf32>
  return
}

// -----

func.func @memref_reinterpret_cast_non_strided_layout(%in: memref<?x?xf32>) {
  // expected-error @+1 {{expected result type to have strided layout but found 'memref<9x10xf32, affine_map<(d0, d1) -> (d0)>>}}
  %out = memref.reinterpret_cast %in to
           offset: [0], sizes: [9, 10], strides: [42, 1]
         : memref<?x?xf32> to memref<9x10xf32, affine_map<(d0, d1) -> (d0)>>
  return
}

// -----

func.func @memref_reshape_element_type_mismatch(
       %buf: memref<*xf32>, %shape: memref<1xi32>) {
  // expected-error @+1 {{element types of source and destination memref types should be the same}}
  memref.reshape %buf(%shape) : (memref<*xf32>, memref<1xi32>) -> memref<?xi32>
}

// -----

func.func @memref_reshape_dst_ranked_shape_unranked(
       %buf: memref<*xf32>, %shape: memref<?xi32>) {
  // expected-error @+1 {{cannot use shape operand with dynamic length to reshape to statically-ranked memref type}}
  memref.reshape %buf(%shape) : (memref<*xf32>, memref<?xi32>) -> memref<?xf32>
}

// -----

func.func @memref_reshape_dst_shape_rank_mismatch(
       %buf: memref<*xf32>, %shape: memref<1xi32>) {
  // expected-error @+1 {{length of shape operand differs from the result's memref rank}}
  memref.reshape %buf(%shape)
    : (memref<*xf32>, memref<1xi32>) -> memref<?x?xf32>
}

// -----

func.func @memref_reshape_src_affine_map_is_not_identity(
        %buf: memref<4x4xf32, strided<[3, 2], offset: 0>>,
        %shape: memref<1xi32>) {
  // expected-error @+1 {{source memref type should have identity affine map}}
  memref.reshape %buf(%shape)
    : (memref<4x4xf32, strided<[3, 2], offset: 0>>, memref<1xi32>)
    -> memref<8xf32>
}

// -----

func.func @memref_reshape_result_affine_map_is_not_identity(
        %buf: memref<4x4xf32>, %shape: memref<1xi32>) {
  // expected-error @+1 {{result memref type should have identity affine map}}
  memref.reshape %buf(%shape)
    : (memref<4x4xf32>, memref<1xi32>) -> memref<8xf32, strided<[2], offset: 0>>
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

func.func @nonexistent_global_memref() {
  // expected-error @+1 {{'gv' does not reference a valid global memref}}
  %0 = memref.get_global @gv : memref<3xf32>
  return
}

// -----

func.func private @foo()

func.func @nonexistent_global_memref() {
  // expected-error @+1 {{'foo' does not reference a valid global memref}}
  %0 = memref.get_global @foo : memref<3xf32>
  return
}

// -----

memref.global @gv : memref<3xi32>

func.func @mismatched_types() {
  // expected-error @+1 {{result type 'memref<3xf32>' does not match type 'memref<3xi32>' of the global memref @gv}}
  %0 = memref.get_global @gv : memref<3xf32>
  return
}

// -----

// expected-error @+1 {{alignment attribute value 63 is not a power of 2}}
memref.global "private" @gv : memref<4xf32> = dense<1.0> { alignment = 63 }

// -----

func.func @copy_different_shape(%arg0: memref<2xf32>, %arg1: memref<3xf32>) {
  // expected-error @+1 {{op requires the same shape for all operands}}
  memref.copy %arg0, %arg1 : memref<2xf32> to memref<3xf32>
  return
}

// -----

func.func @copy_different_eltype(%arg0: memref<2xf32>, %arg1: memref<2xf16>) {
  // expected-error @+1 {{op requires the same element type for all operands}}
  memref.copy %arg0, %arg1 : memref<2xf32> to memref<2xf16>
  return
}

// -----

func.func @expand_shape(%arg0: memref<?x?xf32>) {
  // expected-error @+1 {{invalid number of reassociation groups: found 1, expected 2}}
  %0 = memref.expand_shape %arg0 [[0, 1]] : memref<?x?xf32> into memref<?x5x?xf32>
  return
}

// -----

func.func @expand_shape(%arg0: memref<f32>) {
  // expected-error @+1 {{rank 0 memrefs can only be extended/collapsed with/from ones}}
  %0 = memref.expand_shape %arg0 [] : memref<f32> into memref<1x2xf32>
  return
}

// -----

func.func @collapse_shape_out_of_bounds(%arg0: memref<?x?xf32>) {
  // expected-error @+1 {{op reassociation index 2 is out of bounds}}
  %0 = memref.collapse_shape %arg0 [[0, 1, 2]] : memref<?x?xf32> into memref<?xf32>
}

// -----

func.func @expand_shape_invalid_ranks(%arg0: memref<?x?xf32>) {
  // expected-error @+1 {{op expected rank expansion, but found source rank 2 >= result rank 2}}
  %0 = memref.expand_shape %arg0 [[0], [1]] : memref<?x?xf32> into memref<?x?xf32>
}

// -----

func.func @collapse_shape_invalid_ranks(%arg0: memref<?x?xf32>) {
  // expected-error @+1 {{op expected rank reduction, but found source rank 2 <= result rank 2}}
  %0 = memref.collapse_shape %arg0 [[0], [1]] : memref<?x?xf32> into memref<?x?xf32>
}

// -----

func.func @expand_shape_out_of_bounds(%arg0: memref<?xf32>) {
  // expected-error @+1 {{op reassociation index 2 is out of bounds}}
  %0 = memref.expand_shape %arg0 [[0, 1, 2]] : memref<?xf32> into memref<4x?xf32>
}

// -----

func.func @expand_shape_invalid_result_layout(
    %arg0: memref<30x20xf32, strided<[4000, 2], offset: 100>>) {
  // expected-error @+1 {{expected expanded type to be 'memref<2x15x20xf32, strided<[60000, 4000, 2], offset: 100>>' but found 'memref<2x15x20xf32, strided<[5000, 4000, 2], offset: 100>>'}}
  %0 = memref.expand_shape %arg0 [[0, 1], [2]] :
      memref<30x20xf32, strided<[4000, 2], offset: 100>>
      into memref<2x15x20xf32, strided<[5000, 4000, 2], offset: 100>>
}

// -----

func.func @collapse_shape_mismatch_indices_num(%arg0: memref<?x?x?xf32>) {
  // expected-error @+1 {{invalid number of reassociation groups: found 1, expected 2}}
  %0 = memref.collapse_shape %arg0 [[0, 1]] :
    memref<?x?x?xf32> into memref<?x?xf32, strided<[?, 1], offset: 0>>
}

// -----

func.func @collapse_shape_invalid_reassociation(%arg0: memref<?x?x?xf32>) {
  // expected-error @+1 {{reassociation indices must be contiguous}}
  %0 = memref.collapse_shape %arg0 [[0, 1], [1, 2]] :
    memref<?x?x?xf32> into memref<?x?xf32, strided<[?, 1], offset: 0>>
}

// -----

func.func @collapse_shape_reshaping_non_contiguous(
    %arg0: memref<3x4x5xf32, strided<[270, 50, 10], offset: 0>>) {
  // expected-error @+1 {{invalid source layout map or collapsing non-contiguous dims}}
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] :
      memref<3x4x5xf32, strided<[270, 50, 10], offset: 0>>
      into memref<12x5xf32, strided<[50, 1], offset: 0>>
  return
}

// -----

func.func @collapse_shape_wrong_collapsed_type(%arg0: memref<?x?x?xf32>) {
  // expected-error @+1 {{expected collapsed type to be 'memref<?x?xf32>' but found 'memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>>'}}
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] :
    memref<?x?x?xf32> into memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>>
}

// -----

func.func @expand_shape_illegal_dynamic_memref
  (%arg0: memref<?x?x?xf32>) -> memref<?x?x?x4x?xf32> {
  // expected-error @+1 {{at most one dimension in a reassociation group may be dynamic}}
  %0 = memref.expand_shape %arg0 [[0], [1], [2, 3, 4]]
      : memref<?x?x?xf32> into memref<?x?x?x4x?xf32>
  return %0 : memref<?x?x?x4x?xf32>
}

// -----

func.func @expand_shape_illegal_static_memref
  (%arg0: memref<2x3x20xf32>) -> memref<2x3x2x4x5xf32> {
  // expected-error @+1 {{collapsed dim size (20) must equal reassociation group size (40)}}
  %0 = memref.expand_shape %arg0 [[0], [1], [2, 3, 4]]
      : memref<2x3x20xf32> into memref<2x3x2x4x5xf32>
  return %0 : memref<2x3x2x4x5xf32>
}

// -----

func.func @collapse_shape_illegal_static_memref
  (%arg0: memref<2x3x2x4x5xf32>) -> memref<2x3x20xf32> {
  // expected-error @+1 {{collapsed dim size (20) must equal reassociation group size (40)}}
  %0 = memref.collapse_shape %arg0 [[0], [1], [2, 3, 4]]
      : memref<2x3x2x4x5xf32> into memref<2x3x20xf32>
  return %0 : memref<2x3x20xf32>
}

// -----

func.func @expand_shape_illegal_mixed_memref(%arg0 : memref<?x?xf32>)
    -> memref<?x4x5xf32> {
  // expected-error @+1 {{collapsed dim (1) must be dynamic if and only if reassociation group is dynamic}}
  %0 = memref.expand_shape %arg0 [[0, 1], [2]]
      : memref<?x?xf32> into memref<?x4x5xf32>
  return %0 : memref<?x4x5xf32>
}

// -----

func.func @expand_shape_illegal_mixed_memref_2(%arg0 : memref<?x?xf32>)
    -> memref<?x4x5xf32> {
  // expected-error @+1 {{collapsed dim (1) must be dynamic if and only if reassociation group is dynamic}}
  %0 = memref.expand_shape %arg0 [[0], [1, 2]]
      : memref<?x?xf32> into memref<?x4x5xf32>
  return %0 : memref<?x4x5xf32>
}

// -----

func.func @expand_shape_invalid_static_dim_size(%arg0 : memref<?x21xf32>)
    -> memref<?x4x5xf32> {
  // expected-error @+1 {{collapsed dim size (21) must equal reassociation group size (20)}}
  %0 = memref.expand_shape %arg0 [[0], [1, 2]]
      : memref<?x21xf32> into memref<?x4x5xf32>
  return %0 : memref<?x4x5xf32>
}

// -----

func.func @collapse_shape_illegal_mixed_memref(%arg0 : memref<?x4x5xf32>)
    -> memref<?x?xf32> {
  // expected-error @+1 {{collapsed dim (1) must be dynamic if and only if reassociation group is dynamic}}
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]]
      : memref<?x4x5xf32> into memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// -----

func.func @collapse_shape_illegal_mixed_memref_2(%arg0 : memref<?x4x5xf32>)
    -> memref<?x?xf32> {
  // expected-error @+1 {{collapsed dim (1) must be dynamic if and only if reassociation group is dynamic}}
  %0 = memref.collapse_shape %arg0 [[0], [1, 2]]
      : memref<?x4x5xf32> into memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// -----

func.func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<2048xi8>
  // expected-error@+1 {{expected SSA operand}}
  %1 = memref.view %0[][%arg0, %arg1]
    : memref<2048xi8> to memref<?x?xf32>
  return
}

// -----

func.func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<2048xi8, affine_map<(d0) -> (d0 floordiv 8, d0 mod 8)>>
  // expected-error@+1 {{unsupported map for base memref type}}
  %1 = memref.view %0[%arg2][%arg0, %arg1]
    : memref<2048xi8, affine_map<(d0) -> (d0 floordiv 8, d0 mod 8)>> to
      memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + d1 + s0)>>
  return
}

// -----

func.func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<2048xi8>
  // expected-error@+1 {{unsupported map for result memref type}}
  %1 = memref.view %0[%arg2][%arg0, %arg1]
    : memref<2048xi8> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0, d1, s0)>>
  return
}

// -----

func.func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<2048xi8, 2>
  // expected-error@+1 {{different memory spaces}}
  %1 = memref.view %0[%arg2][%arg0, %arg1] :  memref<2048xi8, 2> to memref<?x?xf32, 1>
  return
}

// -----

func.func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<2048xi8>
  // expected-error@+1 {{incorrect number of size operands for type}}
  %1 = memref.view %0[%arg2][%arg0]
    : memref<2048xi8> to memref<?x?xf32>
  return
}

// -----

func.func @invalid_subview(%input: memref<4x1024xf32>) -> memref<2x256xf32, strided<[1024, 1], offset: 2304>> {
  // expected-error@+1 {{expected offsets to be non-negative, but got -1}}
  %0 = memref.subview %input[-1, 256] [2, 256] [1, 1] : memref<4x1024xf32> to memref<2x256xf32, strided<[1024, 1], offset: 2304>>
  return %0 : memref<2x256xf32, strided<[1024, 1], offset: 2304>>
}

// -----

func.func @invalid_subview(%input: memref<4x1024xf32>) -> memref<2x256xf32, strided<[1024, 1], offset: 2304>> {
  // expected-error@+1 {{expected sizes to be non-negative, but got -1}}
  %0 = memref.subview %input[2, 256] [-1, 256] [1, 1] : memref<4x1024xf32> to memref<2x256xf32, strided<[1024, 1], offset: 2304>>
  return %0 : memref<2x256xf32, strided<[1024, 1], offset: 2304>>
}

// -----

func.func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected mixed offsets rank to match mixed sizes rank (2 vs 3) so the rank of the result type is well-formed}}
  %1 = memref.subview %0[0, 0][2, 2, 2][1, 1, 1]
    : memref<8x16x4xf32> to memref<8x16x4xf32>
  return
}

// -----

func.func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected mixed sizes rank to match mixed strides rank (3 vs 2) so the rank of the result type is well-formed}}
  %1 = memref.subview %0[0, 0, 0][2, 2, 2][1, 1]
    : memref<8x16x4xf32> to memref<8x16x4xf32>
  return
}

// -----

func.func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected mixed sizes rank to match mixed strides rank (3 vs 2) so the rank of the result type is well-formed}}
  %1 = memref.reinterpret_cast %0 to offset: [0], sizes: [2, 2, 2], strides:[1, 1]
    : memref<8x16x4xf32> to memref<8x16x4xf32>
  return
}

// -----

func.func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>, 2>
  // expected-error@+1 {{different memory spaces}}
  %1 = memref.subview %0[0, 0, 0][%arg2, %arg2, %arg2][1, 1, 1]
    : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>, 2> to
      memref<8x?x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * s0 + d1 * 4 + d2)>>
  return
}

// -----

func.func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 + d1, d1 + d2, d2)>>
  // expected-error@+1 {{is not strided}}
  %1 = memref.subview %0[0, 0, 0][%arg2, %arg2, %arg2][1, 1, 1]
    : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 + d1, d1 + d2, d2)>> to
      memref<8x?x4xf32, strided<[?, 4, 1], offset: 0>>
  return
}

// -----

func.func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected 3 offset values}}
  %1 = memref.subview %0[%arg0, %arg1, 0, 0][%arg2, 0, 0, 0][1, 1, 1, 1]
    : memref<8x16x4xf32> to
      memref<8x?x4xf32, strided<[?, ?, 4], offset: 0>>
  return
}

// -----

func.func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected result element type to be 'f32'}}
  %1 = memref.subview %0[0, 0, 0][8, 16, 4][1, 1, 1]
    : memref<8x16x4xf32> to
      memref<8x16x4xi32>
  return
}

// -----

func.func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected result rank to be smaller or equal to the source rank.}}
  %1 = memref.subview %0[0, 0, 0][8, 16, 4][1, 1, 1]
    : memref<8x16x4xf32> to
      memref<8x16x4x3xi32>
  return
}

// -----

func.func @invalid_rank_reducing_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected result type to be 'memref<8x16x4xf32, strided<[64, 4, 1]>>' or a rank-reduced version. (mismatch of result sizes)}}
  %1 = memref.subview %0[0, 0, 0][8, 16, 4][1, 1, 1]
    : memref<8x16x4xf32> to memref<16x4xf32>
  return
}

// -----

func.func @invalid_rank_reducing_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected result type to be 'memref<8x16x4xf32, strided<[64, 4, 1], offset: 8>>' or a rank-reduced version. (mismatch of result sizes)}}
  %1 = memref.subview %0[0, 2, 0][8, 16, 4][1, 1, 1]
    : memref<8x16x4xf32> to memref<16x4xf32>
  return
}

// -----

func.func @invalid_rank_reducing_subview(%arg0 : memref<?x?xf32>, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{expected result type to be 'memref<?x1xf32, strided<[?, 1], offset: ?>>' or a rank-reduced version. (mismatch of result layout)}}
  %0 = memref.subview %arg0[0, %arg1][%arg2, 1][1, 1] : memref<?x?xf32> to memref<?xf32>
  return
}

// -----

#map0 = affine_map<(d0, d1)[s0] -> (d0 * 16 + d1)>

func.func @subview_bad_offset_1(%arg0: memref<16x16xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  // expected-error @+1 {{expected result type to be 'memref<8x8xf32, strided<[16, 1], offset: ?>>' or a rank-reduced version}}
  %s2 = memref.subview %arg0[%c8, %c8][8, 8][1, 1]  : memref<16x16xf32> to memref<8x8xf32, #map0>
  return
}

// -----

#map0 = affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + 136)>

func.func @subview_bad_offset_2(%arg0: memref<16x16xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  // expected-error @+1 {{expected result type to be 'memref<8x8xf32, strided<[16, 1], offset: ?>>' or a rank-reduced version}}
  %s2 = memref.subview %arg0[%c8, 8][8, 8][1, 1]  : memref<16x16xf32> to memref<8x8xf32, #map0>
  return
}

// -----

func.func @subview_bad_offset_3(%arg0: memref<16x16xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  // expected-error @+1 {{expected result type to be 'memref<8x8xf32, strided<[16, 1], offset: ?>>' or a rank-reduced version}}
  %s2 = memref.subview %arg0[%c8, 8][8, 8][1, 1]  : memref<16x16xf32> to memref<8x8xf32, strided<[16, 1], offset: 437>>
  return
}

// -----

func.func @invalid_memref_cast(%arg0 : memref<12x4x16xf32, strided<[64, 16, 1], offset: 0>>) {
  // expected-error@+1{{operand type 'memref<12x4x16xf32, strided<[64, 16, 1]>>' and result type 'memref<12x4x16xf32, strided<[128, 32, 2]>>' are cast incompatible}}
  %0 = memref.cast %arg0 : memref<12x4x16xf32, strided<[64, 16, 1], offset: 0>> to memref<12x4x16xf32, strided<[128, 32, 2], offset: 0>>
  return
}

// -----

func.func @invalid_memref_cast(%arg0 : memref<12x4x16xf32, strided<[64, 16, 1], offset: 0>>) {
  // expected-error@+1{{operand type 'memref<12x4x16xf32, strided<[64, 16, 1]>>' and result type 'memref<12x4x16xf32, strided<[64, 16, 1], offset: 16>>' are cast incompatible}}
  %0 = memref.cast %arg0 : memref<12x4x16xf32, strided<[64, 16, 1], offset: 0>> to memref<12x4x16xf32, strided<[64, 16, 1], offset: 16>>
  return
}

// -----

// incompatible element types
func.func @invalid_memref_cast() {
  %0 = memref.alloc() : memref<2x5xf32, 0>
  // expected-error@+1 {{operand type 'memref<2x5xf32>' and result type 'memref<*xi32>' are cast incompatible}}
  %1 = memref.cast %0 : memref<2x5xf32, 0> to memref<*xi32>
  return
}

// -----

func.func @invalid_prefetch_rw(%i : index) {
  %0 = memref.alloc() : memref<10xf32>
  // expected-error@+1 {{rw specifier has to be 'read' or 'write'}}
  memref.prefetch %0[%i], rw, locality<0>, data  : memref<10xf32>
  return
}

// -----

func.func @invalid_prefetch_cache_type(%i : index) {
  %0 = memref.alloc() : memref<10xf32>
  // expected-error@+1 {{cache type has to be 'data' or 'instr'}}
  memref.prefetch %0[%i], read, locality<0>, false  : memref<10xf32>
  return
}

// -----

func.func @invalid_prefetch_locality_hint(%i : index) {
  %0 = memref.alloc() : memref<10xf32>
  // expected-error@+1 {{32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3}}
  memref.prefetch %0[%i], read, locality<5>, data  : memref<10xf32>
  return
}

// -----

// incompatible memory space
func.func @invalid_memref_cast() {
  %0 = memref.alloc() : memref<2x5xf32, 0>
  // expected-error@+1 {{operand type 'memref<2x5xf32>' and result type 'memref<*xf32, 1>' are cast incompatible}}
  %1 = memref.cast %0 : memref<2x5xf32, 0> to memref<*xf32, 1>
  return
}

// -----

// unranked to unranked
func.func @invalid_memref_cast() {
  %0 = memref.alloc() : memref<2x5xf32, 0>
  %1 = memref.cast %0 : memref<2x5xf32, 0> to memref<*xf32, 0>
  // expected-error@+1 {{operand type 'memref<*xf32>' and result type 'memref<*xf32>' are cast incompatible}}
  %2 = memref.cast %1 : memref<*xf32, 0> to memref<*xf32, 0>
  return
}

// -----

// alignment is not power of 2.
func.func @assume_alignment(%0: memref<4x4xf16>) {
  // expected-error@+1 {{alignment must be power of 2}}
  memref.assume_alignment %0, 12 : memref<4x4xf16>
  return
}

// -----

// 0 alignment value.
func.func @assume_alignment(%0: memref<4x4xf16>) {
  // expected-error@+1 {{attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive}}
  memref.assume_alignment %0, 0 : memref<4x4xf16>
  return
}

// -----

"alloca_without_scoped_alloc_parent"() ({
  memref.alloca() : memref<1xf32>
  // expected-error@-1 {{requires an ancestor op with AutomaticAllocationScope trait}}
  func.return
}) : () -> ()

// -----

func.func @bad_alloc_wrong_dynamic_dim_count() {
^bb0:
  %0 = arith.constant 7 : index
  // Test alloc with wrong number of dynamic dimensions.
  // expected-error@+1 {{dimension operand count does not equal memref dynamic dimension count}}
  %1 = memref.alloc(%0)[%0] : memref<2x4xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
  return
}

// -----

func.func @bad_alloc_wrong_symbol_count() {
^bb0:
  %0 = arith.constant 7 : index
  // Test alloc with wrong number of symbols
  // expected-error@+1 {{symbol operand count does not equal memref symbol count}}
  %1 = memref.alloc(%0) : memref<2x?xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
  return
}

// -----

func.func @load_invalid_memref_indexes() {
  %0 = memref.alloca() : memref<10xi32>
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{incorrect number of indices for load, expected 1 but got 2}}
  %1 = memref.load %0[%c0, %c0] : memref<10xi32>
}

// -----

func.func @test_store_zero_results() {
^bb0:
  %0 = memref.alloc() : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>
  %1 = arith.constant 0 : index
  %2 = arith.constant 1 : index
  %3 = memref.load %0[%1, %2] : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>
  // Test that store returns zero results.
  %4 = memref.store %3, %0[%1, %2] : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1> // expected-error {{cannot name an operation with no results}}
  return
}

// -----

func.func @test_store_zero_results2(%x: i32, %p: memref<i32>) {
  "memref.store"(%x,%p) : (i32, memref<i32>) -> i32  // expected-error {{'memref.store' op requires zero results}}
  return
}

// -----

func.func @test_alloc_memref_map_rank_mismatch() {
^bb0:
  // expected-error@+1 {{memref layout mismatch between rank and affine map: 2 != 1}}
  %0 = memref.alloc() : memref<1024x64xf32, affine_map<(d0) -> (d0)>, 1>
  return
}

// -----

func.func @rank(%0: f32) {
  // expected-error@+1 {{'memref.rank' op operand #0 must be ranked or unranked memref of any type values}}
  "memref.rank"(%0): (f32)->index
  return
}

// -----

#map = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (s0 + d0 * s1 + d1 * s2 + d2 * s3)>
func.func @illegal_num_offsets(%arg0 : memref<?x?x?xf32>, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{expected 3 offset values}}
  %0 = memref.subview %arg0[0, 0] [%arg1, %arg2] [1, 1] : memref<?x?x?xf32> to memref<?x?x?xf32, #map>
}

// -----

func.func @atomic_rmw_idxs_rank_mismatch(%I: memref<16x10xf32>, %i : index, %val : f32) {
  // expected-error@+1 {{expects the number of subscripts to be equal to memref rank}}
  %x = memref.atomic_rmw addf %val, %I[%i] : (f32, memref<16x10xf32>) -> f32
  return
}

// -----

func.func @atomic_rmw_expects_float(%I: memref<16x10xi32>, %i : index, %val : i32) {
  // expected-error@+1 {{expects a floating-point type}}
  %x = memref.atomic_rmw addf %val, %I[%i, %i] : (i32, memref<16x10xi32>) -> i32
  return
}

// -----

func.func @atomic_rmw_expects_int(%I: memref<16x10xf32>, %i : index, %val : f32) {
  // expected-error@+1 {{expects an integer type}}
  %x = memref.atomic_rmw addi %val, %I[%i, %i] : (f32, memref<16x10xf32>) -> f32
  return
}

// -----

func.func @generic_atomic_rmw_wrong_arg_num(%I: memref<10xf32>, %i : index) {
  // expected-error@+1 {{expected single number of entry block arguments}}
  %x = memref.generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%arg0 : f32, %arg1 : f32):
      %c1 = arith.constant 1.0 : f32
      memref.atomic_yield %c1 : f32
  }
  return
}

// -----

func.func @generic_atomic_rmw_wrong_arg_type(%I: memref<10xf32>, %i : index) {
  // expected-error@+1 {{expected block argument of the same type result type}}
  %x = memref.generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%old_value : i32):
      %c1 = arith.constant 1.0 : f32
      memref.atomic_yield %c1 : f32
  }
  return
}

// -----

func.func @generic_atomic_rmw_result_type_mismatch(%I: memref<10xf32>, %i : index) {
 // expected-error@+1 {{failed to verify that result type matches element type of memref}}
 %0 = "memref.generic_atomic_rmw"(%I, %i) ({
    ^bb0(%old_value: f32):
      %c1 = arith.constant 1.0 : f32
      memref.atomic_yield %c1 : f32
    }) : (memref<10xf32>, index) -> i32
  return
}

// -----

func.func @generic_atomic_rmw_has_side_effects(%I: memref<10xf32>, %i : index) {
  // expected-error@+4 {{should contain only operations with no side effects}}
  %x = memref.generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%old_value : f32):
      %c1 = arith.constant 1.0 : f32
      %buf = memref.alloc() : memref<2048xf32>
      memref.atomic_yield %c1 : f32
  }
}

// -----

func.func @atomic_yield_type_mismatch(%I: memref<10xf32>, %i : index) {
  // expected-error@+4 {{op types mismatch between yield op: 'i32' and its parent: 'f32'}}
  %x = memref.generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%old_value : f32):
      %c1 = arith.constant 1 : i32
      memref.atomic_yield %c1 : i32
  }
  return
}

// -----

#map0 = affine_map<(d0) -> (d0 floordiv 8, d0 mod 8)>
func.func @memref_realloc_layout(%src : memref<256xf32, #map0>) -> memref<?xf32>{
  // expected-error@+1 {{unsupported layout}}
  %0 = memref.realloc %src : memref<256xf32, #map0> to memref<?xf32>
  return %0 : memref<?xf32>
}

// -----

func.func @memref_realloc_sizes_1(%src : memref<2xf32>) -> memref<?xf32>{
  // expected-error@+1 {{missing dimension operand}}
  %0 = memref.realloc %src : memref<2xf32> to memref<?xf32>
  return %0 : memref<?xf32>
}

// -----

func.func @memref_realloc_sizes_2(%src : memref<?xf32>, %d : index)
  -> memref<4xf32>{
  // expected-error@+1 {{unnecessary dimension operand}}
  %0 = memref.realloc %src(%d) : memref<?xf32> to memref<4xf32>
  return %0 : memref<4xf32>
}

// -----

func.func @memref_realloc_type(%src : memref<256xf32>) -> memref<?xi32>{
  // expected-error@+1 {{different element types}}
  %0 = memref.realloc %src : memref<256xf32> to memref<?xi32>
  return %0 : memref<?xi32>
}

// -----

// Asking the dimension of a 0-D shape doesn't make sense.
func.func @dim_0_ranked(%arg : memref<f32>, %arg1 : index) {
  memref.dim %arg, %arg1 : memref<f32> // expected-error {{'memref.dim' op operand #0 must be unranked.memref of any type values or non-0-ranked.memref of any type values, but got 'memref<f32>'}}
  return
}

// -----

func.func @subview_invalid_strides(%m: memref<7x22x333x4444xi32>) {
  // expected-error @below{{expected result type to be 'memref<7x11x333x4444xi32, strided<[32556744, 2959704, 4444, 1]>>' or a rank-reduced version. (mismatch of result layout)}}
  %subview = memref.subview %m[0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1]
      : memref<7x22x333x4444xi32> to memref<7x11x333x4444xi32>
  return
}

// -----

func.func @subview_invalid_strides_rank_reduction(%m: memref<7x22x333x4444xi32>) {
  // expected-error @below{{expected result type to be 'memref<7x11x1x4444xi32, strided<[32556744, 2959704, 4444, 1]>>' or a rank-reduced version. (mismatch of result layout)}}
  %subview = memref.subview %m[0, 0, 0, 0] [7, 11, 1, 4444] [1, 2, 1, 1]
      : memref<7x22x333x4444xi32> to memref<7x11x4444xi32>
  return
}
