// RUN: mlir-opt <%s -split-input-file -verify-diagnostics

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
  // expected-error@+2 {{'result' must be 1D tensor of any type values, but got 'tensor<*xi32>'}}
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
  // expected-error @+2 {{op expects regions to end with 'tensor.yield', found 'std.return'}}
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
