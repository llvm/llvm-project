// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{expected a non-empty array for level types}}
#a = #sparse_tensor.encoding<{dimLevelType = []}>
func.func private @scalar(%arg0: tensor<f64, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>
func.func private @tensor_dimlevel_size_mismatch(%arg0: tensor<8xi32, #a>) -> () // expected-error {{expected an array of size 1 for dimension level types}}

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
func.func private @tensor_dimorder_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{higherOrdering = "wrong"}> // expected-error {{expected an affine map for higher ordering}}
func.func private @tensor_highorder_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

// expected-error@+1 {{expected a permutation affine map for dimension ordering}}
#a = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"], dimOrdering = affine_map<(i,j) -> (i,i)>}>
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

// -----

#a = #sparse_tensor.encoding<{dimLevelType = [ "compressed", "compressed", "dense", "dense" ], dimOrdering  = affine_map<(ii, jj, i, j) -> (ii, jj, i, j)>, higherOrdering = affine_map<(i, j) -> (j, i)>}> // expected-error {{unexpected higher ordering mapping from 2 to 2}}
func.func private @tensor_invalid_key(%arg0: tensor<10x60xf32, #a>) -> ()

// -----

#CSR_SLICE = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  slice = [ (-1, ?, 1), (?, 4, 2) ] // expected-error{{expect positive value or ? for slice offset/size/stride}}
}>
func.func private @sparse_slice(tensor<?x?xf64, #CSR_SLICE>)
