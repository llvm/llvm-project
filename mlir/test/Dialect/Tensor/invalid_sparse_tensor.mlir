// RUN: mlir-opt <%s -split-input-file -verify-diagnostics

// -----

#a = #tensor.sparse<{sparseDimLevelType = [1,2]}>
func private @tensor_size_mismatch(%arg0: tensor<8xi32, #a>) -> () // expected-error {{expected an array of size 1 for dimension level types}}

// -----

#a = #tensor.sparse<{sparseDimLevelType = [1]}>
func private @tensor_type_mismatch(%arg0: tensor<8xi32, #a>) -> () // expected-error {{expected string value in dimension level types}}

// -----

#a = #tensor.sparse<{sparseDimLevelType = ["strange"]}>
func private @tensor_value_mismatch(%arg0: tensor<8xi32, #a>) -> () // expected-error {{unexpected dimension level type: "strange"}}

// -----

#a = #tensor.sparse<{sparseDimOrdering = "wrong"}>
func private @tensor_order_mismatch(%arg0: tensor<8xi32, #a>) -> () // expected-error {{expected an affine map for dimension ordering}}

// -----

#a = #tensor.sparse<{sparseDimOrdering = affine_map<(i,j) -> (i,i)>}>
func private @tensor_no_permutation(%arg0: tensor<16x32xf32, #a>) -> () // expected-error {{expected a permutation affine map of size 2 for dimension ordering}}

// -----

#a = #tensor.sparse<{sparsePointerBitWidth = 42}>
func private @tensor_invalid_int_ptr(%arg0: tensor<16x32xf32, #a>) -> () // expected-error {{unexpected bitwidth: 42}}

// -----

#a = #tensor.sparse<{sparseIndexBitWidth = "not really"}>
func private @tensor_no_int_index(%arg0: tensor<16x32xf32, #a>) -> () // expected-error {{expected an integral bitwidth}}

// -----

#a = #tensor.sparse<{sparseIndexBitWidth = 128}>
func private @tensor_invalid_int_index(%arg0: tensor<16x32xf32, #a>) -> () // expected-error {{unexpected bitwidth: 128}}

// -----

#a = #tensor.sparse<{key = 1}>
func private @tensor_invalid_key(%arg0: tensor<16x32xf32, #a>) -> () // expected-error {{unexpected key: key}}
