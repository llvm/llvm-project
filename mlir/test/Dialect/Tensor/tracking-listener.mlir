// RUN: mlir-opt -test-tensor-transform-patterns=test-tracking-listener \
// RUN:     -split-input-file -verify-diagnostics %s

func.func @replace_op_with_op_of_same_type() {
  %0 = "test.foo"() {replaced} : () -> (tensor<5xf32>)
  // expected-remark @below {{replacement found}}
  %1 = "test.foo"() {replacement_0 = 0} : () -> (tensor<5xf32>)
  return
}

// -----

func.func @replace_op_with_op_of_different_type() {
  // expected-error @below {{listener could not find replacement op}}
  %0 = tensor.empty() {replaced} : tensor<5xf32>
  %1 = "test.foo"() {replacement_0 = 0} : () -> (tensor<5xf32>)
  return
}

// -----

func.func @multi_result_replacement() {
  %0:2 = "test.foo"() {replaced} : () -> (tensor<5xf32>, tensor<6xf32>)
  // expected-remark @below {{replacement found}}
  %1:2 = "test.foo"() {replacement_0 = 0, replacement_1 = 1}
      : () -> (tensor<5xf32>, tensor<6xf32>)
  return
}

// -----

func.func @multi_result_replacement_with_multiple_ops() {
  // expected-error @below {{listener could not find replacement op}}
  %0:2 = "test.foo"() {replaced} : () -> (tensor<5xf32>, tensor<6xf32>)
  %1:2 = "test.foo"() {replacement_0 = 0} : () -> (tensor<5xf32>, tensor<6xf32>)
  %2:2 = "test.foo"() {replacement_1 = 1} : () -> (tensor<5xf32>, tensor<6xf32>)
  return
}

// -----

func.func @replacement_wrapped_in_cast() {
  %0 = "test.foo"() {replaced} : () -> (tensor<5xf32>)
  // expected-remark @below {{replacement found}}
  %1 = "test.foo"() : () -> (tensor<?xf32>)
  %2 = tensor.cast %1 {replacement_0 = 0} : tensor<?xf32> to tensor<5xf32>
  return
}

// -----

func.func @replacement_wrapped_in_chain_of_casts() {
  %0 = "test.foo"() {replaced} : () -> (tensor<5xf32>)
  // expected-remark @below {{replacement found}}
  %1 = "test.foo"() : () -> (tensor<?xf32>)
  %2 = tensor.cast %1 : tensor<?xf32> to tensor<5xf32>
  %3 = tensor.cast %2 : tensor<5xf32> to tensor<?xf32>
  %4 = tensor.cast %3 {replacement_0 = 0} : tensor<?xf32> to tensor<5xf32>
  return
}

// -----

func.func @cast_like_insert_slice(%t: tensor<1x5xf32>) {
  %0 = "test.foo"() {replaced} : () -> (tensor<5xf32>)
  // expected-remark @below {{replacement found}}
  %1 = "test.foo"() : () -> (tensor<5xf32>)
  %2 = tensor.insert_slice %1 into %t[0, 0][1, 5][1, 1] {replacement_0 = 0}
      : tensor<5xf32> into tensor<1x5xf32>
  return
}

// -----

func.func @non_cast_like_insert_slice(%t: tensor<7xf32>) {
  // expected-error @below {{listener could not find replacement op}}
  %0 = "test.foo"() {replaced} : () -> (tensor<5xf32>)
  %1 = "test.foo"() : () -> (tensor<5xf32>)
  // This is not a cast-like insert_slice op because elements from %t are
  // contained in %2.
  %2 = tensor.insert_slice %1 into %t[0][5][1] {replacement_0 = 0}
      : tensor<5xf32> into tensor<7xf32>
  return
}

// -----

func.func @cast_like_insert_slice_dynamic(
    %t: tensor<1x?x1xf32>, %f: f32, %pos: index) {
  %c0 = arith.constant 0 : index
  %0 = tensor.insert %f into %t[%c0, %pos, %c0] {replaced} : tensor<1x?x1xf32>

  // Rank reduction
  %c1 = arith.constant 1 : index
  %dim1 = tensor.dim %t, %c1 : tensor<1x?x1xf32>
  %1 = tensor.extract_slice %t[0, 0, 0][1, %dim1, 1][1, 1, 1]
      : tensor<1x?x1xf32> to tensor<?xf32>
  // expected-remark @below {{replacement found}}
  %2 = tensor.insert %f into %1[%c0] : tensor<?xf32>
  // Rank expansion
  // Throw in a wrench: Do not use %dim1 directly, but another SSA value that
  // has the same runtime value.
  %dim1b = tensor.dim %1, %c0 : tensor<?xf32>
  %3 = tensor.insert_slice %2 into %t[0, 0, 0][1, %dim1b, 1][1, 1, 1]
      {replacement_0 = 0} : tensor<?xf32> into tensor<1x?x1xf32>
  return
}

// -----

func.func @cast_like_extract_slice() {
  %0 = "test.foo"() {replaced} : () -> (tensor<5xf32>)
  // expected-remark @below {{replacement found}}
  %1 = "test.foo"() : () -> (tensor<1x5x1x1xf32>)
  %2 = tensor.extract_slice %1[0, 0, 0, 0][1, 5, 1, 1][1, 1, 1, 1]
      {replacement_0 = 0} : tensor<1x5x1x1xf32> to tensor<5xf32>
  return
}

// -----

func.func @cast_like_extract_slice_dynamic() {
  %0 = "test.foo"() {replaced} : () -> (tensor<?xf32>)
  // expected-remark @below {{replacement found}}
  %1 = "test.foo"() : () -> (tensor<1x?x1x1xf32>)
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %1, %c1 : tensor<1x?x1x1xf32>
  %2 = tensor.extract_slice %1[0, 0, 0, 0][1, %dim, 1, 1][1, 1, 1, 1]
      {replacement_0 = 0} : tensor<1x?x1x1xf32> to tensor<?xf32>
  return
}

// -----

func.func @non_cast_like_extract_slice() {
  // expected-error @below {{listener could not find replacement op}}
  %0 = "test.foo"() {replaced} : () -> (tensor<5xf32>)
  %1 = "test.foo"() : () -> (tensor<1x5x1x1xf32>)
  %2 = tensor.extract_slice %1[0, 0, 0, 0][1, 3, 1, 1][1, 1, 1, 1]
      {replacement_0 = 0} : tensor<1x5x1x1xf32> to tensor<3xf32>
  return
}

// -----

func.func @non_cast_like_extract_slice_drop_non_unit_dim() {
  // expected-error @below {{listener could not find replacement op}}
  %0 = "test.foo"() {replaced} : () -> (tensor<f32>)
  %1 = "test.foo"() : () -> (tensor<1x5x1x1xf32>)
  %2 = tensor.extract_slice %1[0, 0, 0, 0][1, 1, 1, 1][1, 1, 1, 1]
      {replacement_0 = 0} : tensor<1x5x1x1xf32> to tensor<f32>
  return
}
