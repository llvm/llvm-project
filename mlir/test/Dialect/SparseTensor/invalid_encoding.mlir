// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{expected a non-empty array for lvlTypes}}
#a = #sparse_tensor.encoding<{lvlTypes = []}>
func.func private @scalar(%arg0: tensor<f64, #a>) -> ()

// -----

// expected-error@+2 {{dimension-rank mismatch between encoding and tensor shape: 2 != 1}}
#a = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>
func.func private @tensor_dimlevel_size_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

// expected-error@+1 {{use of undeclared identifier}}
#a = #sparse_tensor.encoding<{map = (d0) -> (d0 : dense, d1 : compressed)}>
func.func private @tensor_sizes_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{lvlTypes = [1]}> // expected-error {{expected a string value in lvlTypes}}
func.func private @tensor_type_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{lvlTypes = ["strange"]}> // expected-error {{unexpected level-type: strange}}
func.func private @tensor_value_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{dimToLvl = "wrong"}> // expected-error {{expected an affine map for dimToLvl}}
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

// expected-error@+1 {{expected a permutation affine map for dimToLvl}}
#a = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d0 : compressed)}>
func.func private @tensor_no_permutation(%arg0: tensor<16x32xf32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{posWidth = "x"}> // expected-error {{expected an integral position bitwidth}}
func.func private @tensor_no_int_ptr(%arg0: tensor<16x32xf32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{posWidth = 42}> // expected-error {{unexpected position bitwidth: 42}}
func.func private @tensor_invalid_int_ptr(%arg0: tensor<16x32xf32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{crdWidth = "not really"}> // expected-error {{expected an integral index bitwidth}}
func.func private @tensor_no_int_index(%arg0: tensor<16x32xf32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{crdWidth = 128}> // expected-error {{unexpected coordinate bitwidth: 128}}
func.func private @tensor_invalid_int_index(%arg0: tensor<16x32xf32, #a>) -> ()

// -----

#a = #sparse_tensor.encoding<{key = 1}> // expected-error {{unexpected key: key}}
func.func private @tensor_invalid_key(%arg0: tensor<16x32xf32, #a>) -> ()

// -----

#CSR_SLICE = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  dimSlices = [ (-1, ?, 1), (?, 4, 2) ] // expected-error{{expect positive value or ? for slice offset/size/stride}}
}>
func.func private @sparse_slice(tensor<?x?xf64, #CSR_SLICE>)

// -----

// expected-error@+2 {{Level-rank mismatch between forward-declarations and specifiers. Declared 3 level-variables; but got 2 level-specifiers.}}
#TooManyLvlDecl = #sparse_tensor.encoding<{
  map = {l0, l1, l2} (d0, d1) -> (l0 = d0 : dense, l1 = d1 : compressed)
}>
func.func private @too_many_lvl_decl(%arg0: tensor<?x?xf64, #TooManyLvlDecl>) {
  return
}

// -----

// expected-error@+2 {{use of undeclared identifier 'l1'}}
#TooFewLvlDecl = #sparse_tensor.encoding<{
  map = {l0} (d0, d1) -> (l0 = d0 : dense, l1 = d1 : compressed)
}>
func.func private @too_few_lvl_decl(%arg0: tensor<?x?xf64, #TooFewLvlDecl>) {
  return
}

// -----

// expected-error@+2 {{Level-variable ordering mismatch. The variable 'l0' was forward-declared as the 1st level; but is bound by the 0th specification.}}
#WrongOrderLvlDecl = #sparse_tensor.encoding<{
  map = {l1, l0} (d0, d1) -> (l0 = d0 : dense, l1 = d1 : compressed)
}>
func.func private @wrong_order_lvl_decl(%arg0: tensor<?x?xf64, #WrongOrderLvlDecl>) {
  return
}
