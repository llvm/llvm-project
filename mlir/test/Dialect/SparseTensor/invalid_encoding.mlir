// RUN: mlir-opt %s -split-input-file -verify-diagnostics

#a = #sparse_tensor.encoding<{dimLevelType = []}>
func.func private @scalar(%arg0: tensor<f64, #a>) -> () // expected-error {{expected non-scalar sparse tensor}}

// -----

#a = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>
func.func private @tensor_size_mismatch(%arg0: tensor<8xi32, #a>) -> () // expected-error {{expected an array of size 1 for dimension level types}}

// -----

#a = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"], dimOrdering = affine_map<(i) -> (i)>}> // expected-error {{unexpected mismatch in ordering and dimension level types size}}
func.func private @tensor_sizes_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{dimLevelType = [1]}> // expected-error {{expected a string value in dimension level types}}
func.func private @tensor_type_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{dimLevelType = ["strange"]}> // expected-error {{unexpected dimension level type: strange}}
func.func private @tensor_value_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{dimOrdering = "wrong"}> // expected-error {{expected an affine map for dimension ordering}}
func.func private @tensor_order_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{dimOrdering = affine_map<(i,j) -> (i,i)>}> // expected-error {{expected a permutation affine map for dimension ordering}}
func.func private @tensor_no_permutation(%arg0: tensor<16x32xf32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{pointerBitWidth = "x"}> // expected-error {{expected an integral pointer bitwidth}}
func.func private @tensor_no_int_ptr(%arg0: tensor<16x32xf32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{pointerBitWidth = 42}> // expected-error {{unexpected pointer bitwidth: 42}}
func.func private @tensor_invalid_int_ptr(%arg0: tensor<16x32xf32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{indexBitWidth = "not really"}> // expected-error {{expected an integral index bitwidth}}
func.func private @tensor_no_int_index(%arg0: tensor<16x32xf32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{indexBitWidth = 128}> // expected-error {{unexpected index bitwidth: 128}}
func.func private @tensor_invalid_int_index(%arg0: tensor<16x32xf32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{key = 1}> // expected-error {{unexpected key: key}}
func.func private @tensor_invalid_key(%arg0: tensor<16x32xf32, #a>) -> ()
