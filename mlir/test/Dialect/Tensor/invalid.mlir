// RUN: mlir-opt <%s -split-input-file -verify-diagnostics

func @dim(%arg : tensor<1x?xf32>) {
  %c2 = arith.constant 2 : index
  tensor.dim %arg, %c2 : tensor<1x?xf32> // expected-error {{'tensor.dim' op index is out of range}}
  return
}

// -----

func @tensor.cast_mismatching_constants(%arg0: tensor<1xf32>) {
  // expected-error@+1 {{operand type 'tensor<1xf32>' and result type 'tensor<2xf32>' are cast incompatible}}
  %0 = tensor.cast %arg0 : tensor<1xf32> to tensor<2xf32>
  return
}

// -----

func @extract_too_many_indices(%arg0: tensor<?xf32>) {
  // expected-error@+1 {{incorrect number of indices for extract_element}}
  %0 = tensor.extract %arg0[] : tensor<?xf32>
  return
}

// -----

func @insert_too_many_indices(%arg0: f32, %arg1: tensor<?xf32>) {
  // expected-error@+1 {{incorrect number of indices}}
  %0 = tensor.insert %arg0 into %arg1[] : tensor<?xf32>
  return
}

// -----

func @tensor.from_elements_wrong_result_type() {
  // expected-error@+2 {{'result' must be statically shaped tensor of any type values, but got 'tensor<*xi32>'}}
  %c0 = arith.constant 0 : i32
  %0 = tensor.from_elements %c0 : tensor<*xi32>
  return
}

// -----

func @tensor.from_elements_wrong_elements_count() {
  // expected-error@+2 {{1 operands present, but expected 2}}
  %c0 = arith.constant 0 : index
  %0 = tensor.from_elements %c0 : tensor<2xindex>
  return
}

// -----

func @tensor.generate(%m : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{must have as many index operands as dynamic extents in the result type}}
  %tnsr = tensor.generate %m {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = arith.constant 8.0 : f32
      tensor.yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func @tensor.generate(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{must have one body argument per input dimension}}
  %tnsr = tensor.generate %m, %n {
    ^bb0(%i : index, %j : index):
      %elem = arith.constant 8.0 : f32
      tensor.yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func @tensor.generate(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{all body arguments must be index}}
  %tnsr = tensor.generate %m, %n {
    ^bb0(%i : index, %j : index, %k : i64):
      %elem = arith.constant 8.0 : f32
      tensor.yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func @tensor.generate(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+2 {{op expects regions to end with 'tensor.yield', found 'func.return'}}
  // expected-note @+1 {{in custom textual format, the absence of terminator implies 'tensor.yield'}}
  %tnsr = tensor.generate %m, %n {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = arith.constant 8.0 : f32
      return %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func @tensor.generate(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{body must be terminated with a `yield` operation of the tensor element type}}
  %tnsr = tensor.generate %m, %n {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = arith.constant 8 : i32
      tensor.yield %elem : i32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}
// -----

func @tensor.reshape_element_type_mismatch(
       %buf: tensor<*xf32>, %shape: tensor<1xi32>) {
  // expected-error @+1 {{element types of source and destination tensor types should be the same}}
  tensor.reshape %buf(%shape) : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xi32>
}

// -----

func @tensor.reshape_dst_ranked_shape_unranked(
       %buf: tensor<*xf32>, %shape: tensor<?xi32>) {
  // expected-error @+1 {{cannot use shape operand with dynamic length to reshape to statically-ranked tensor type}}
  tensor.reshape %buf(%shape) : (tensor<*xf32>, tensor<?xi32>) -> tensor<?xf32>
}

// -----

func @tensor.reshape_dst_shape_rank_mismatch(
       %buf: tensor<*xf32>, %shape: tensor<1xi32>) {
  // expected-error @+1 {{length of shape operand differs from the result's tensor rank}}
  tensor.reshape %buf(%shape)
    : (tensor<*xf32>, tensor<1xi32>) -> tensor<?x?xf32>
}

// -----

func @tensor.reshape_num_elements_mismatch(
       %buf: tensor<1xf32>, %shape: tensor<1xi32>) {
  // expected-error @+1 {{source and destination tensor should have the same number of elements}}
  tensor.reshape %buf(%shape)
    : (tensor<1xf32>, tensor<1xi32>) -> tensor<10xf32>
}

// -----

func @extract_slice_wrong_result_rank(%t: tensor<?xf32>, %idx : index) {
  // expected-error @+1 {{expected rank to be smaller or equal to the other rank.}}
  %0 = tensor.extract_slice %t[0][4][1] : tensor<?xf32> to tensor<?x?xf32>

  return
}

// -----

func @extract_slice_wrong_result_rank(%t: tensor<?xf32>, %idx : index) {
  // expected-error @+1 {{expected element type to be 'f32'}}
  %0 = tensor.extract_slice %t[0][4][1] : tensor<?xf32> to tensor<4xi8>

  return
}

// -----

func @extract_slice_wrong_static_type(%t: tensor<8x16x4xf32>, %idx : index) {
  // expected-error @+1 {{expected type to be 'tensor<?x4x4xf32>' or a rank-reduced version. (size mismatch)}}
  %0 = tensor.extract_slice %t[0, 0, 0][%idx, 4, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<4x4x4xf32>

  return
}

// -----

func @extract_slice_wrong_dynamic_type(%t: tensor<8x16x4xf32>, %idx : index) {
  // expected-error @+1 {{expected type to be 'tensor<4x4x4xf32>' or a rank-reduced version. (size mismatch)}}
  %0 = tensor.extract_slice %t[0, 2, 0][4, 4, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<?x4x4xf32>

  return
}

// -----

func @insert_slice_wrong_result_rank(%t1: tensor<?xf32>, %t2: tensor<?x?xf32>, %idx : index) {
  // expected-error @+1 {{expected rank to be smaller or equal to the other rank.}}
  %0 = tensor.insert_slice %t2 into %t1[0][4][1] : tensor<?x?xf32> into tensor<?xf32>

  return
}

// -----

func @insert_slice_wrong_result_rank(%t1: tensor<4xi8>, %t2: tensor<?xf32>, %idx : index) {
  // expected-error @+1 {{expected element type to be 'f32'}}
  %0 = tensor.insert_slice %t1 into %t2[0][4][1] : tensor<4xi8> into tensor<?xf32>

  return
}

// -----

func @insert_slice_wrong_static_type(%t1: tensor<4x4x4xf32>, %t2: tensor<8x16x4xf32>, %idx : index) {
  // expected-error @+1 {{expected type to be 'tensor<?x4x4xf32>' or a rank-reduced version. (size mismatch)}}
  %0 = tensor.insert_slice %t1 into %t2[0, 0, 0][%idx, 4, 4][1, 1, 1]
    : tensor<4x4x4xf32> into tensor<8x16x4xf32>

  return
}

// -----

func @insert_slice_wrong_dynamic_type(%t1: tensor<?x4x4xf32>, %t2: tensor<8x16x4xf32>, %idx : index) {
  // expected-error @+1 {{expected type to be 'tensor<4x4x4xf32>' or a rank-reduced version. (size mismatch)}}
  %0 = tensor.insert_slice %t1 into %t2[0, 2, 0][4, 4, 4][1, 1, 1]
    : tensor<?x4x4xf32> into tensor<8x16x4xf32>

  return
}

// -----

func @illegal_expanding_reshape_dynamic_tensor
  (%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?x4x?xf32> {
  // expected-error @+1 {{invalid to have a single dimension (2) expanded into multiple dynamic dims (2,4)}}
  %0 = tensor.expand_shape %arg0 [[0], [1], [2, 3, 4]]
      : tensor<?x?x?xf32> into tensor<?x?x?x4x?xf32>
  return %0 : tensor<?x?x?x4x?xf32>
}

// -----


func @illegal_expanding_reshape_static_tensor
    (%arg0: tensor<2x3x20xf32>) -> tensor<2x3x2x4x5xf32> {
  // expected-error @+1 {{expected dimension 2 of collapsed type to be static value of 40}}
  %0 = tensor.expand_shape %arg0 [[0], [1], [2, 3, 4]]
      : tensor<2x3x20xf32> into tensor<2x3x2x4x5xf32>
  return %0 : tensor<2x3x2x4x5xf32>
}

// -----

func @illegal_collapsing_reshape_static_tensor
    (%arg0: tensor<2x3x2x4x5xf32>) -> tensor<2x3x20xf32> {
  // expected-error @+1 {{expected dimension 2 of collapsed type to be static value of 40}}
  %0 = tensor.collapse_shape %arg0 [[0], [1], [2, 3, 4]]
      : tensor<2x3x2x4x5xf32> into tensor<2x3x20xf32>
  return %0 : tensor<2x3x20xf32>
}

// -----

func @illegal_expanding_reshape_mixed_tensor(%arg0 : tensor<?x?xf32>)
    -> tensor<?x4x5xf32> {
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 5}}
  %0 = tensor.expand_shape %arg0 [[0, 1], [2]]
      : tensor<?x?xf32> into tensor<?x4x5xf32>
  return %0 : tensor<?x4x5xf32>
}

// -----

func @illegal_expanding_reshape_mixed_tensor_2(%arg0 : tensor<?x?xf32>)
    -> tensor<?x4x5xf32> {
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 20}}
  %0 = tensor.expand_shape %arg0 [[0], [1, 2]]
      : tensor<?x?xf32> into tensor<?x4x5xf32>
  return %0 : tensor<?x4x5xf32>
}

// -----

func @illegal_collapsing_reshape_mixed_tensor(%arg0 : tensor<?x4x5xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 5}}
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2]]
      : tensor<?x4x5xf32> into tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @illegal_collapsing_reshape_mixed_tensor_2(%arg0 : tensor<?x4x5xf32>)
    -> tensor<?x?xf32> {
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 20}}
  %0 = tensor.collapse_shape %arg0 [[0], [1, 2]]
      : tensor<?x4x5xf32> into tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @rank(%0: f32) {
  // expected-error@+1 {{'tensor.rank' op operand #0 must be tensor of any type values}}
  "tensor.rank"(%0): (f32)->index
  return
}

// -----

func @illegal_num_offsets(%arg0 : tensor<?x?x?xf32>, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{expected 3 offset values}}
  %0 = tensor.extract_slice %arg0[0, 0] [%arg1, %arg2] [1, 1] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
  return
}

// -----

func @illegal_num_offsets(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?x?xf32>,
    %arg2 : index, %arg3 : index) {
  // expected-error@+1 {{expected 3 offset values}}
  %0 = tensor.insert_slice %arg0 into %arg1[0, 0] [%arg2, %arg3] [1, 1] : tensor<?x?xf32> into tensor<?x?x?xf32>
  return
}

// -----


func @pad_result_type(%arg0: tensor<?x2x3x4xi32>, %arg1: index, %arg2: i32) -> tensor<?x?x?x8xf32> {
  // expected-error @+1 {{specified type 'tensor<?x?x?x8xf32>' does not match the inferred type 'tensor<?x?x?x9xi32>}}
  %0 = tensor.pad %arg0 low[1, %arg1, 2, 2] high[1, 2, %arg1, 3] {
  ^bb0(%arg3: index, %arg4: index):
    tensor.yield %arg2 : i32
  } : tensor<?x2x3x4xi32> to tensor<?x?x?x8xf32>
  return %0 : tensor<?x?x?x8xf32>
}

// -----

func @pad_number_of_block_args(%arg0: tensor<?x4xi32>, %arg1: i32) -> tensor<?x9xi32> {
  // expected-error @+1 {{expected the block to have 2 arguments}}
  %0 = tensor.pad %arg0 low[1, 2] high[2, 3] {
  ^bb0(%arg2: index, %arg3: index, %arg4: index):
    tensor.yield %arg1 : i32
  } : tensor<?x4xi32> to tensor<?x9xi32>
  return %0 : tensor<?x9xi32>
}

// -----

func @pad_block_args(%arg0: tensor<?x4xi32>, %arg1: i32) -> tensor<?x9xi32> {
  // expected-error @+1 {{op expected block argument 1 to be an index}}
  %0 = tensor.pad %arg0 low[1, 2] high[2, 3] {
  ^bb0(%arg2: i32, %arg3: i32):
    tensor.yield %arg1 : i32
  } : tensor<?x4xi32> to tensor<?x9xi32>
  return %0 : tensor<?x9xi32>
}

// -----

func @pad_yield_type(%arg0: tensor<?x4xi32>, %arg1: i8) -> tensor<?x9xi32> {
  // expected-error @+1 {{op expected yield type to match shape element type}}
  %0 = tensor.pad %arg0 low[1, 2] high[2, 3] {
  ^bb0(%arg2: index, %arg3: index):
    tensor.yield %arg1 : i8
  } : tensor<?x4xi32> to tensor<?x9xi32>
  return %0 : tensor<?x9xi32>
}

// -----

func @invalid_splat(%v : f32) {
  // expected-error@+1 {{invalid kind of type specified}}
  tensor.splat %v : memref<8xf32>
  return
}

// -----

func @invalid_splat(%v : vector<8xf32>) {
  // expected-error@+1 {{must be integer/index/float type}}
  %w = tensor.splat %v : tensor<8xvector<8xf32>>
  return
}