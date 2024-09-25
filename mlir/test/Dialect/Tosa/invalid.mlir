// RUN: mlir-opt %s -split-input-file -verify-diagnostics --tosa-validate=strict-op-spec-alignment


func.func @test_const() -> tensor<1xf32> {
  // expected-error@+1{{'tosa.const' op expected same attr/result element types}}
  %0 = "tosa.const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

func.func @test_const_non_tensor_attr() {
  // expected-error@+1{{tosa.const' op expected tensors for attr/result type}}
  %0 = "tosa.const"() {value = dense<1.0> : vector<f32>} : () -> tensor<f32>
  return
}

// -----

func.func @test_conv2d(%arg0: tensor<1x29x29x4xf32>, %arg1: tensor<16x3x3x4xi8>, %arg2: tensor<16xi8>) -> tensor<1x27x27x16xi8> {
  // expected-error@+1 {{expect both input and weight to be float or not together, got 'f32' and 'i8'}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
           : (tensor<1x29x29x4xf32>, tensor<16x3x3x4xi8>, tensor<16xi8>) -> tensor<1x27x27x16xi8>
  return %0 : tensor<1x27x27x16xi8>
}

// -----

func.func @test_conv2d(%arg0: tensor<*xi8>, %arg1: tensor<16x3x3x4xi8>, %arg2: tensor<16xi8>) -> tensor<1x27x27x16xi8> {
  // expected-error@+1 {{expect a ranked tensor for input, got <block argument> of type 'tensor<*xi8>' at index: 0}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
           : (tensor<*xi8>, tensor<16x3x3x4xi8>, tensor<16xi8>) -> tensor<1x27x27x16xi8>
  return %0 : tensor<1x27x27x16xi8>
}

// -----

func.func @test_conv2d(%arg0: tensor<1x29x29x4xi8>, %arg1: tensor<*xi8>, %arg2: tensor<16xi8>) -> tensor<1x27x27x16xi8> {
  // expected-error@+1 {{'tosa.conv2d' op operand #1 must be 4D tensor of 4-bit signless integer or 8-bit signless integer or Quint8 type or Qint4 type or Qint8 type or Qint16 type or Qint32 type or floating-point values, but got 'tensor<*xi8>'}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
           : (tensor<1x29x29x4xi8>, tensor<*xi8>, tensor<16xi8>) -> tensor<1x27x27x16xi8>
  return %0 : tensor<1x27x27x16xi8>
}

// -----

func.func @test_conv2d(%arg0: tensor<1x29x29x4xi8>, %arg1: tensor<16x3x3x4xi8>, %arg2: tensor<16xi8>) -> tensor<1x27x27x16xi8> {
  // expected-error@+1 {{'tosa.conv2d' op quantizationattr is required for quantized type, and not allowed for float type}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
           : (tensor<1x29x29x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi8>) -> tensor<1x27x27x16xi8>
  return %0 : tensor<1x27x27x16xi8>
}

// -----

func.func @test_concat(%arg0 : tensor<2x1xf32>, %arg1 : tensor<2x2xf32>) -> tensor<?x?xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Cannot concat tensors with different sizes on the non-axis dimension 1}}
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @test_concat_element_type_mismatch(%arg0 : tensor<1x2xf32>, %arg1 : tensor<2x2xf32>) -> tensor<?x?xi8> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'tosa.concat' op inferred type(s) 'tensor<3x2xf32>' are incompatible with return type(s) of operation 'tensor<?x?xi8>}}
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<?x?xi8>
  return %0 : tensor<?x?xi8>
}

// -----

func.func @test_pad_non_const(%arg0: tensor<13x21x3xf32>, %arg1: tensor<3x2xi32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.pad' op padding of pad is not constant}}
  %0 = tosa.pad %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

func.func @test_pad_non_const(%arg0: tensor<13x21x3xi8>, %arg1: tensor<i8>) -> tensor<13x21x3xi8> {
  %0 = "tosa.const"() {value = dense<[[0, 0], [0, 1], [0, 1]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // expected-error@+1 {{'tosa.pad' op pad_const of pad is not constant}}
  %1 = tosa.pad %arg0, %0, %arg1 : (tensor<13x21x3xi8>, tensor<3x2xi32>, tensor<i8>) -> tensor<13x21x3xi8>
  return %1 : tensor<13x21x3xi8>
}

// -----

func.func @test_pad_io_rank_mismatch(%arg0: tensor<13x21xf32>, %arg1: tensor<2x2xi32>) {
  // expected-error@+1 {{'tosa.pad' op expect same input and output tensor rank.}}
  %1 = tosa.pad %arg0, %arg1 : (tensor<13x21xf32>, tensor<2x2xi32>) -> tensor<13x21x3xf32>
  return
}

// -----

func.func @test_pad_invalid_padding_rank(%arg0: tensor<13x21xf32>, %arg1: tensor<2xi32>) {
  // expected-error@+1 {{'tosa.pad' op expect 'padding' tensor rank equal to 2.}}
  %1 = tosa.pad %arg0, %arg1 : (tensor<13x21xf32>, tensor<2xi32>) -> tensor<13x21xf32>
  return
}

// -----

func.func @test_pad_invalid_padConst_rank(%arg0: tensor<13x21xf32>, %arg1: tensor<2x2xi32>) {
  %0 = "tosa.const"() {value = dense<3.14> : tensor<1xf32>} : () -> tensor<1xf32>
  // expected-error@+1 {{'tosa.pad' op operand #2 must be 0D tensor of number values, but got 'tensor<1xf32>'}}
  %1 = tosa.pad %arg0, %arg1, %0 : (tensor<13x21xf32>, tensor<2x2xi32>, tensor<1xf32>) -> tensor<13x21xf32>
  return
}

// -----

func.func @test_transpose_non_const(%arg0: tensor<13x21x3xf32>, %arg1: tensor<3xi32>) -> tensor<3x13x21xf32> {
  // expected-error@+1 {{'tosa.transpose' op perms of transpose is not constant}}
  %0 = tosa.transpose %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21xf32>
  return %0 : tensor<3x13x21xf32>
}

// -----

func.func @test_transpose_io_rank_mismatch(%arg0: tensor<13x21x3xf32>, %arg1: tensor<3xi32>) -> tensor<3x13x21x1xf32> {
  // expected-error@+1 {{'tosa.transpose' op expected input tensor rank to equal result tensor rank}}
  %0 = tosa.transpose %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21x1xf32>
  return %0 : tensor<3x13x21x1xf32>
}

// -----

func.func @test_transpose_invalid_perms_rank(%arg0: tensor<13x21x3xf32>, %arg1: tensor<3x2xi32>) -> tensor<3x13x21xf32> {
  // expected-error@+1 {{'tosa.transpose' op expected permutation tensor to be rank 1 but got rank 2}}
  %0 = tosa.transpose %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<3x13x21xf32>
  return %0 : tensor<3x13x21xf32>
}

// -----

func.func @test_transpose_rank0_perms() {
  %14 = tensor.empty() : tensor<5x27xi64>
  %cst = tensor.empty() : tensor<i32>
  // expected-error@+1 {{'tosa.transpose' op expected permutation tensor to be rank 1 but got rank 0}}
  %72 = tosa.transpose %14, %cst : (tensor<5x27xi64>, tensor<i32>) -> tensor<?x?xi64>
  return
}

// -----

func.func @test_transpose_invalid_perms_size(%arg0: tensor<13x21x3xf32>, %arg1: tensor<7xi32>) -> tensor<3x13x21xf32> {
  // expected-error@+1 {{'tosa.transpose' op expected permutation tensor dim 0 to have size 3 (input rank) but got size 7}}
  %0 = tosa.transpose %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<7xi32>) -> tensor<3x13x21xf32>
  return %0 : tensor<3x13x21xf32>
}

// -----

func.func @test_transpose_invalid_permutation_tensor(%arg0: tensor<13x21x3xf32>) -> tensor<?x?x?xf32> {
  %perms = arith.constant dense<[2, 0, 0]> : tensor<3xi32>
  // expected-error@+1 {{'tosa.transpose' op expected valid permutation tensor}}
  %0 = tosa.transpose %arg0, %perms : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

func.func @test_transpose_invalid_permutation_negative(%arg0: tensor<3x2xi32>) -> tensor<*xi32> {
  %perms = "tosa.const"() {value = dense<[-1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error@+1 {{'tosa.transpose' op expected valid permutation tensor}}
  %1 = tosa.transpose %arg0, %perms : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<*xi32>
  return %1 : tensor<*xi32>
}

// -----

func.func @test_transpose_invalid_permutation_tensor_above_range(%arg0: tensor<3x2xi32>) -> tensor<*xi32> {
  %perms = "tosa.const"() {value = dense<[2, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error@+1 {{'tosa.transpose' op expected valid permutation tensor}}
  %1 = tosa.transpose %arg0, %perms : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<*xi32>
  return %1 : tensor<*xi32>
}

// -----

func.func @test_transpose_invalid_permutation_types(%arg0: tensor<3x2xi32>) -> tensor<3x4xi32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error@+1 {{'tosa.transpose' op expected output tensor dim 0 to match input dim 1 with value of 2}}
  %1 = tosa.transpose %arg0, %perms : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<3x4xi32>
  return %1 : tensor<3x4xi32>
}

// -----

func.func @test_transpose_invalid_permutation_types_dynamic_dim_ok(%arg0: tensor<2x?xi32>) -> tensor<3x4xi32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error@+1 {{'tosa.transpose' op expected output tensor dim 1 to match input dim 0 with value of 2}}
  %1 = tosa.transpose %arg0, %perms : (tensor<2x?xi32>, tensor<2xi32>) -> tensor<3x4xi32>
  return %1 : tensor<3x4xi32>
}

// -----

func.func @test_fully_connected_non_const(%arg0: tensor<13x21x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<273x2xf32> {
  %0 = "tosa.const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %1 = tosa.reshape %arg0 {new_shape = array<i64: 273, 3>} : (tensor<13x21x3xf32>) -> tensor<273x3xf32>
  // expected-error@+1 {{'tosa.fully_connected' op weight of fully_connected is not constant}}
  %2 = tosa.fully_connected %1, %arg1, %0 : (tensor<273x3xf32>, tensor<2x3xf32>, tensor<2xf32>) -> tensor<273x2xf32>
  return %2 : tensor<273x2xf32>
}

// -----

func.func @test_fully_connected_non_const(%arg0: tensor<13x21x3xf32>, %arg1: tensor<2xf32>) -> tensor<273x2xf32> {
  %0 = "tosa.const"() {value = dense<[[-0.613216758, -0.63714242, -0.73500061], [0.180762768, 0.773053169, -0.933686495]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %1 = tosa.reshape %arg0 {new_shape = array<i64: 273, 3>} : (tensor<13x21x3xf32>) -> tensor<273x3xf32>
  // expected-error@+1 {{'tosa.fully_connected' op bias of fully_connected is not constant}}
  %2 = tosa.fully_connected %1, %0, %arg1 : (tensor<273x3xf32>, tensor<2x3xf32>, tensor<2xf32>) -> tensor<273x2xf32>
  return %2 : tensor<273x2xf32>
}

// -----

func.func @test_reduce_sum_type_mismatch(%arg0 : tensor<2x3x4x5xf32>) -> () {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'tosa.reduce_sum' op inferred type(s) 'tensor<1x3x4x5xf32>' are incompatible with return type(s) of operation 'tensor<1x3x4x5xi32>'}}
  %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<2x3x4x5xf32>) -> tensor<1x3x4x5xi32>
  return
}

// -----

func.func @test_reduce_max_type_mismatch(%arg0 : tensor<2x3x4x5xf32>) -> () {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'tosa.reduce_max' op inferred type(s) 'tensor<2x3x4x1xf32>' are incompatible with return type(s) of operation 'tensor<2x3x4x1xi32>'}}
  %0 = tosa.reduce_max %arg0 {axis = 3 : i32} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x1xi32>
  return
}

// -----

func.func @test_reduce_min_type_mismatch(%arg0 : tensor<2x3x4x5xf32>) -> () {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'tosa.reduce_min' op inferred type(s) 'tensor<2x1x4x5xf32>' are incompatible with return type(s) of operation 'tensor<2x1x4x5xi32>'}}
  %0 = tosa.reduce_min %arg0 {axis = 1 : i32} : (tensor<2x3x4x5xf32>) -> tensor<2x1x4x5xi32>
  return
}

// -----

func.func @test_reduce_prod_type_mismatch(%arg0 : tensor<2x3x4x5xf32>) -> () {
  // expected-error@+1 {{'tosa.reduce_prod' op expect reduced dimension size to be 1, got 3}}
  %0 = tosa.reduce_prod %arg0 {axis = 1 : i32} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
  return
}

// -----

func.func @test_reduce_all_invalid_axis(%arg0 : tensor<2x3x4xf32>) -> () {
  // expected-error@+1 {{'tosa.reduce_all' op expect input tensor rank (3) to be larger than reduce axis (3)}}
  %0 = tosa.reduce_all %arg0 {axis = 3 : i32} : (tensor<2x3x4xf32>) -> tensor<2x3x1xf32>
  return
}

// -----

func.func @test_reduce_any_invalid_axis(%arg0 : tensor<2x3x4xf32>) -> () {
  // expected-error@+1 {{'tosa.reduce_any' op expect input tensor rank (3) to be larger than reduce axis (3)}}
  %0 = tosa.reduce_any %arg0 {axis = 3 : i32} : (tensor<2x3x4xf32>) -> tensor<2x3x1xf32>
  return
}

// -----

func.func @test_reduce_max_invalid_axis(%arg0 : tensor<2x3x4xf32>) -> () {
  // expected-error@+1 {{'tosa.reduce_max' op expect input tensor rank (3) to be larger than reduce axis (3)}}
  %0 = tosa.reduce_max %arg0 {axis = 3 : i32} : (tensor<2x3x4xf32>) -> tensor<2x3x1xf32>
  return
}

// -----

func.func @test_reduce_min_invalid_axis(%arg0 : tensor<2x3x4xf32>) -> () {
  // expected-error@+1 {{'tosa.reduce_min' op expect input tensor rank (3) to be larger than reduce axis (3)}}
  %0 = tosa.reduce_min %arg0 {axis = 3 : i32} : (tensor<2x3x4xf32>) -> tensor<2x3x1xf32>
  return
}

// -----

func.func @test_reduce_prod_invalid_axis(%arg0 : tensor<2x3x4xf32>) -> () {
  // expected-error@+1 {{'tosa.reduce_prod' op expect input tensor rank (3) to be larger than reduce axis (3)}}
  %0 = tosa.reduce_prod %arg0 {axis = 3 : i32} : (tensor<2x3x4xf32>) -> tensor<2x3x1xf32>
  return
}

// -----

func.func @test_reduce_sum_invalid_axis(%arg0 : tensor<2x3x4xf32>) -> () {
  // expected-error@+1 {{'tosa.reduce_sum' op expect input tensor rank (3) to be larger than reduce axis (3)}}
  %0 = tosa.reduce_sum %arg0 {axis = 3 : i32} : (tensor<2x3x4xf32>) -> tensor<2x3x1xf32>
  return
}

// -----

func.func @test_reduce_min_invalid_output_rank(%arg0 : tensor<i32>) -> () {
  // expected-error@+1 {{'tosa.reduce_min' op expect output tensor rank to be equal to input tensor rank}}
  %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<i32>) -> tensor<1x10xi32>
  return
}

// -----

func.func @test_reshape_type_mismatch(%arg0 : tensor<13x21x3xf32>) -> () {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'tosa.reshape' op inferred type(s) 'tensor<13x21x3x1xf32>' are incompatible with return type(s) of operation 'tensor<13x21x3x1xi32>'}}
  %0 = tosa.reshape %arg0 {new_shape = array<i64: 13, 21, 3, 1>} : (tensor<13x21x3xf32>) -> tensor<13x21x3x1xi32>
  return
}

// -----

func.func @test_reshape_static_zero_dim_input(%arg0 : tensor<13x0x3xf32>) -> () {
  // expected-error@+1 {{'tosa.reshape' op operand #0 must be tosa-conformant tensor of number values, but got 'tensor<13x0x3xf32>'}}
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 13, 21, 3>} : (tensor<13x0x3xf32>) -> tensor<13x0x3xf32>
  return
}

// -----

func.func @test_reshape_zero_dim_input(%arg0 : tensor<?x0x3xf32>) -> () {
  // expected-error@+1 {{'tosa.reshape' op operand #0 must be tosa-conformant tensor of number values, but got 'tensor<?x0x3xf32>'}}
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 13, 21, 3>} : (tensor<?x0x3xf32>) -> tensor<13x0x3xf32>
  return
}

// -----

func.func @test_reshape_rank_mismatch(%arg0 : tensor<?xf32>) -> () {
  // expected-error@+1 {{'tosa.reshape' op new shape does not match result rank}}
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 2, 4>} : (tensor<?xf32>) -> tensor<?xf32>
  return
}

// -----

func.func @test_reshape_inconsistent_result_type(%arg0 : tensor<?xf32>) -> () {
  // expected-error@+1 {{'tosa.reshape' op new shape is inconsistent with result shape}}
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 2, 4, -1>} : (tensor<?xf32>) -> tensor<?x3x5xf32>
  return
}

// -----

func.func @test_reshape_invalid_size(%arg0 : tensor<2x4xf32>) -> () {
  // expected-error@+1 {{'tosa.reshape' op cannot reshape 8 elements into 15}}
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 3, 5>} : (tensor<2x4xf32>) -> tensor<3x5xf32>
  return
}

// -----

func.func @test_reshape_invalid_newshape(%arg0 : tensor<1xf32>) -> () {
  // expected-error@+1 {{'tosa.reshape' op cannot reshape 1 elements into 4}}
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: -1, 4>} : (tensor<1xf32>) -> tensor<?x4xf32>
  return
}

// -----

func.func @test_reshape_invalid_newshape(%arg0 : tensor<8xf32>) -> () {
  // expected-error@+1 {{'tosa.reshape' op cannot reshape 8 elements into 4}}
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 4>} : (tensor<8xf32>) -> tensor<?x4xf32>
  return
}

// -----

func.func @test_reshape_invalid_placeholders(%arg0 : tensor<?xf32>) -> () {
  // expected-error@+1 {{'tosa.reshape' op expected at most one target dimension to be -1}}
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 2, -1, -1>} : (tensor<?xf32>) -> tensor<2x?x?xf32>
  return
}

// -----

func.func @test_reshape_invalid_tensor_dim(%arg0 : tensor<4x?xf32>) -> () {
  // expected-error@+1 {{'tosa.reshape' op new shape has invalid tensor dimension size -2}}
  %0 = "tosa.reshape" (%arg0) {new_shape = array<i64: -2, -1>} : (tensor<4x?xf32>) -> tensor<?x4xf32>
  return
}

// -----

func.func @test_reverse_axis_out_of_range(%arg0 : tensor<13x21x3xf32>) -> () {
  // expected-error@+1 {{'tosa.reverse' op expect input tensor rank (3) to be larger than reverse axis (5)}}
  %0 = tosa.reverse %arg0 {axis = 5 : i32} : (tensor<13x21x3xf32>) -> tensor<?x?x?xi32>
  return
}

// -----

func.func @test_const_attribute_type_mismatch() -> tensor<100x100xf32> {
  // expected-error@+1 {{'tosa.const' op failed to verify that all of {value, output} have same shape}}
  %0 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x1xf32>} : () -> tensor<100x100xf32>
  return %0 : tensor<100x100xf32>
}

// -----

func.func @test_conv2d_static_zero_dim_input(%arg0: tensor<1x29x0x4xf32>, %arg1: tensor<16x3x3x4xf32>, %arg2: tensor<16xf32>) -> tensor<1x27x27x16xf32> {
  // expected-error@+1 {{'tosa.conv2d' op operand #0 must be 4-d tosa-conformant tensor, but got 'tensor<1x29x0x4xf32>'}}
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
           : (tensor<1x29x0x4xf32>, tensor<16x3x3x4xf32>, tensor<16xf32>) -> tensor<1x27x27x16xf32>
  return %0 : tensor<1x27x27x16xf32>
}

// -----

func.func @test_conv2d_zero_dim_input(%arg0: tensor<1x?x0x4xf32>, %arg1: tensor<16x3x3x4xf32>, %arg2: tensor<16xf32>) -> tensor<1x27x27x16xf32> {
  // expected-error@+1 {{'tosa.conv2d' op operand #0 must be 4-d tosa-conformant tensor, but got 'tensor<1x?x0x4xf32>'}}
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
           : (tensor<1x?x0x4xf32>, tensor<16x3x3x4xf32>, tensor<16xf32>) -> tensor<1x27x27x16xf32>
  return %0 : tensor<1x27x27x16xf32>
}


// -----

func.func @test_avg_pool2d_static_zero_dim_input(%arg0: tensor<1x0x7x9xf32>) -> tensor<1x7x7x9xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op operand #0 must be 4-d tosa-conformant tensor, but got 'tensor<1x0x7x9xf32>'}}
    %0 = "tosa.avg_pool2d"(%arg0) {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>}
      : (tensor<1x0x7x9xf32>) -> tensor<1x7x7x9xf32>
    return %0 : tensor<1x7x7x9xf32>
}

// -----

func.func @test_avg_pool2d_zero_dim_input(%arg0: tensor<1x0x?x9xf32>) -> tensor<1x7x7x9xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op operand #0 must be 4-d tosa-conformant tensor, but got 'tensor<1x0x?x9xf32>'}}
    %0 = "tosa.avg_pool2d"(%arg0) {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>}
      : (tensor<1x0x?x9xf32>) -> tensor<1x7x7x9xf32>
    return %0 : tensor<1x7x7x9xf32>
}

// -----

func.func @test_variable_duplicates(%arg0: tensor<2x4x8xi32>) -> () {
  tosa.variable @stored_var = dense<-1> : tensor<2x4x8xi32>
  // expected-error@+1 {{'tosa.variable' op name has already been declared}}
  tosa.variable @stored_var : tensor<1x4x8xi32>
  return
}

// -----

func.func @test_variable_read_type(%arg0: tensor<2x4x8xi32>) -> () {
  tosa.variable @stored_var = dense<-1> : tensor<2x4x8xi32>
  // expected-error@+1 {{'tosa.variable.read' op result type does not equal variable type}}
  %0 = tosa.variable.read @stored_var : tensor<2x4x8xi16>
  return
}

// -----

func.func @test_variable_read_shape(%arg0: tensor<2x4x8xi32>) -> () {
  tosa.variable @stored_var = dense<-1> : tensor<2x4x8xi32>
  // expected-error@+1 {{'tosa.variable.read' op result type does not equal variable type}}
  %0 = tosa.variable.read @stored_var : tensor<1x4x8xi32>
  return
}

// -----

func.func @test_variable_write_type(%arg0: tensor<2x4x8xi16>) -> () {
  tosa.variable @stored_var = dense<-1> : tensor<2x4x8xi32>
  // expected-error@+1 {{'tosa.variable.write' op operand type does not equal variable type}}
  tosa.variable.write @stored_var, %arg0 : tensor<2x4x8xi16>
  return
}

// -----

func.func @test_variable_write_shape(%arg0: tensor<1x4x8xi32>) -> () {
  tosa.variable @stored_var = dense<-1> : tensor<2x4x8xi32>
  // expected-error@+1 {{'tosa.variable.write' op operand type does not equal variable type}}
  tosa.variable.write @stored_var, %arg0 : tensor<1x4x8xi32>
  return
}

// -----

func.func @test_slice_invalid_start() {
  %0 = tensor.empty() : tensor<4x31x31xf32>
  // expected-error@+1 {{'tosa.slice' op length of start attribute is not equal rank of input shape}}
  %1 = tosa.slice %0 {size = array<i64: 1, 1, 1>, start = array<i64: 1, 1>} : (tensor<4x31x31xf32>) -> tensor<*xf32>
  return
}

// -----

func.func @test_slice_invalid_size() {
  %0 = tensor.empty() : tensor<4x31x31xf32>
  // expected-error@+1 {{'tosa.slice' op length of size attribute is not equal rank of input shape}}
  %1 = tosa.slice %0 {size = array<i64: 1>, start = array<i64: 1, 1, 1>} : (tensor<4x31x31xf32>) -> tensor<*xf32>
  return
}

// -----

func.func @test_tile_invalid_multiples() {
  %0 = tensor.empty() : tensor<4x31x31xf32>
  // expected-error@+1 {{'tosa.tile' op expect 'multiples' array to have length 3 but got 0.}}
  %1 = tosa.tile %0 {multiples = array<i64>} : (tensor<4x31x31xf32>) -> tensor<4x31x31xf32>
  return
}

// -----

func.func @test_tile_invalid_multiples_value() {
  %0 = tensor.empty() : tensor<4x31xf32>
  // expected-error@+1 {{'tosa.tile' op expect element of 'multiples' to be positive integer or -1.}}
  %1 = tosa.tile %0 {multiples = array<i64: 2, -2>} : (tensor<4x31xf32>) -> tensor<4x31xf32>
  return
}

// -----

func.func @test_tile_io_rank_mismatch() {
  %0 = tensor.empty() : tensor<4x31xf32>
  // expected-error@+1 {{'tosa.tile' op expect same input and output tensor rank.}}
  %1 = tosa.tile %0 {multiples = array<i64: 2, 2>} : (tensor<4x31xf32>) -> tensor<4x31x31xf32>
  return
}

// -----

// CHECK-LABEL: @test_invalid_constant_permutation
func.func @test_invalid_constant_permutation() {
  // expected-error@+3 {{'tosa.transpose' op expected valid permutation tensor}}
  %0 = tensor.empty() : tensor<3x4x5xi32>
  %1 = arith.constant dense<[3, 0, 1]> : tensor<3xi32>
  %2 = tosa.transpose %0, %1 : (tensor<3x4x5xi32>, tensor<3xi32>) -> tensor<3x4x5xi32>
  return
}

// -----

// CHECK-LABEL: test_rank_size_constant_permutation
func.func @test_rank_size_constant_permutation() {
  // expected-error@+4 {{'tosa.transpose' op expected valid permutation tensor}}
  %0 = arith.constant 6 : index
  %1 = arith.constant dense<[0, 2]> : tensor<2xi32>
  %2 = tensor.empty(%0) : tensor<?x27xi64>
  %3 = tosa.transpose %2, %1 : (tensor<?x27xi64>, tensor<2xi32>) -> tensor<?x27xi64>
  return
}

// -----

// CHECK-LABEL: test_large_constant_permutation
func.func @test_large_constant_permutation() {
  // expected-error@+4 {{'tosa.transpose' op expected valid permutation tensor}}
  %0 = arith.constant 6 : index
  %1 = arith.constant dense<[1185677355, 332462212]> : tensor<2xi32>
  %2 = tensor.empty(%0) : tensor<?x27xi64>
  %3 = tosa.transpose %2, %1 : (tensor<?x27xi64>, tensor<2xi32>) -> tensor<?x27xi64>
  return
}

// -----

// CHECK-LABEL: test_table_rank0_table
func.func @test_table_rank0_table(%arg0: tensor<64xi16>, %arg1: tensor<i16>) {
  // expected-error@+1 {{'tosa.table' op operand #1 must be 1-d tosa-conformant tensor, but got 'tensor<i16>'}}
  %0 = tosa.table %arg0, %arg1 : (tensor<64xi16>, tensor<i16>) -> tensor<64xi16>
  return
}

// -----

// CHECK-LABEL: test_table_io_rank_mismatch
func.func @test_table_io_rank_mismatch(%arg0: tensor<64xi16>, %arg1: tensor<6xi16>) {
  // expected-error@+1 {{'tosa.table' op expected input tensor rank to equal result tensor rank}}
  %0 = tosa.table %arg0, %arg1 : (tensor<64xi16>, tensor<6xi16>) -> tensor<64x?xi16>
  return
}

// -----

// CHECK-LABEL: test_table_io_shape_mismatch
func.func @test_table_io_shape_mismatch(%arg0: tensor<?x16xi16>, %arg1: tensor<6xi16>) {
  // expected-error@+1 {{'tosa.table' op dim(result, 1) = 15 doesn't match dim(input, 1) = 16}}
  %0 = tosa.table %arg0, %arg1 : (tensor<?x16xi16>, tensor<6xi16>) -> tensor<?x15xi16>
  return
}

// -----

// CHECK-LABEL: test_transpose_conv2d_invalid_outshape
func.func @test_transpose_conv2d_invalid_outshape(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op attribute 'out_shape' failed to satisfy constraint: i64 dense array attribute with exactly 4 elements}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}
