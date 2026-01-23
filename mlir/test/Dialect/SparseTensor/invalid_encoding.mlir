// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{expected '(' in dimension-specifier list}}
#a = #sparse_tensor.encoding<{map = []}>
func.func private @scalar(%arg0: tensor<f64, #a>) -> ()

// -----

// expected-error@+1 {{expected '->'}}
#a = #sparse_tensor.encoding<{map = ()}>
func.func private @scalar(%arg0: tensor<f64, #a>) -> ()

// -----

// expected-error@+1 {{expected ')' in dimension-specifier list}}
#a = #sparse_tensor.encoding<{map = (d0 -> d0)}>
func.func private @scalar(%arg0: tensor<f64, #a>) -> ()

// -----

// expected-error@+1 {{expected '(' in dimension-specifier list}}
#a = #sparse_tensor.encoding<{map = d0 -> d0}>
func.func private @scalar(%arg0: tensor<f64, #a>) -> ()

// -----

// expected-error@+1 {{expected '(' in level-specifier list}}
#a = #sparse_tensor.encoding<{map = (d0) -> d0}>
func.func private @scalar(%arg0: tensor<f64, #a>) -> ()

// -----

// expected-error@+1 {{expected ':'}}
#a = #sparse_tensor.encoding<{map = (d0) -> (d0)}>
func.func private @scalar(%arg0: tensor<f64, #a>) -> ()

// -----

// expected-error@+1 {{expected valid level format (e.g. dense, compressed or singleton)}}
#a = #sparse_tensor.encoding<{map = (d0) -> (d0:)}>
func.func private @scalar(%arg0: tensor<f64, #a>) -> ()

// -----

// expected-error@+1 {{expected valid level format (e.g. dense, compressed or singleton)}}
#a = #sparse_tensor.encoding<{map = (d0) -> (d0 : (compressed))}>
func.func private @scalar(%arg0: tensor<f64, #a>) -> ()

// -----

// expected-error@+2 {{dimension-rank mismatch between encoding and tensor shape: 2 != 1}}
#a = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>
func.func private @tensor_dimlevel_size_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

// expected-error@+1 {{Batch lvlType can only be leading levels}}
#a = #sparse_tensor.encoding<{map = (d0, d1, d2) -> (d0 : batch, d1 : compressed, d2: batch)}>
func.func private @non_leading_batch(%arg0: tensor<?x?x?i32, #a>) -> ()

// -----

// expected-error@+1 {{use of undeclared identifier}}
#a = #sparse_tensor.encoding<{map = (d0) -> (d0 : dense, d1 : compressed)}>
func.func private @tensor_sizes_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

// expected-error@+1 {{failed to infer lvlToDim from dimToLvl}}
#a = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense)}>
func.func private @tensor_sizes_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

// expected-error@+1 {{expected bare identifier}}
#a = #sparse_tensor.encoding<{map = (1)}>
func.func private @tensor_type_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

// expected-error@+1 {{unexpected key: nap}}
#a = #sparse_tensor.encoding<{nap = (d0) -> (d0 : dense)}>
func.func private @tensor_type_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

// expected-error@+1 {{expected '(' in dimension-specifier list}}
#a = #sparse_tensor.encoding<{map =  -> (d0 : dense)}>
func.func private @tensor_type_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

// expected-error@+1 {{unknown level format: strange}}
#a = #sparse_tensor.encoding<{map = (d0) -> (d0 : strange)}>
func.func private @tensor_value_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

// expected-error@+1 {{expected valid level format (e.g. dense, compressed or singleton)}}
#a = #sparse_tensor.encoding<{map = (d0) -> (d0 : "wrong")}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

// expected-error@+1 {{expected valid level property (e.g. nonordered, nonunique or high)}}
#a = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed("wrong"))}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----
// expected-error@+1 {{expected ')' in level-specifier list}}
#a = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed[high])}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

// expected-error@+1 {{unknown level property: wrong}}
#a = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed(wrong))}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

// expected-error@+1 {{use of undeclared identifier}}
#a = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed, dense)}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<8xi32, #a>) -> ()

// -----

// expected-error@+1 {{failed to infer lvlToDim from dimToLvl}}
#a = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d0 : compressed)}>
func.func private @tensor_no_permutation(%arg0: tensor<16x32xf32, #a>) -> ()

// -----

// expected-error@+1 {{unexpected character}}
#a = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : compressed; d1 : dense)}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<16x32xi32, #a>) -> ()

// -----

// expected-error@+1 {{expected attribute value}}
#a = #sparse_tensor.encoding<{map = (d0: d1) -> (d0 : compressed, d1 : dense)}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<16x32xi32, #a>) -> ()

// -----

// expected-error@+1 {{expected ':'}}
#a = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 = compressed, d1 = dense)}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<16x32xi32, #a>) -> ()

// -----

// expected-error@+1 {{expected attribute value}}
#a = #sparse_tensor.encoding<{map = (d0 : compressed, d1 : compressed)}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<16x32xi32, #a>) -> ()

// -----

// expected-error@+1 {{use of undeclared identifier}}
#a = #sparse_tensor.encoding<{map = (d0 = compressed, d1 = compressed)}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<16x32xi32, #a>) -> ()

// -----

// expected-error@+1 {{use of undeclared identifier}}
#a = #sparse_tensor.encoding<{map = (d0 = l0, d1 = l1) {l0, l1} -> (l0 = d0 : dense, l1 = d1 : compressed)}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<16x32xi32, #a>) -> ()

// -----

// expected-error@+1 {{expected '='}}
#a = #sparse_tensor.encoding<{map = {l0, l1} (d0 = l0, d1 = l1) -> (l0 : d0 = dense, l1 : d1 = compressed)}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<16x32xi32, #a>) -> ()

// -----
// expected-error@+1 {{use of undeclared identifier 'd0'}}
#a = #sparse_tensor.encoding<{map = {l0, l1} (d0 = l0, d1 = l1) -> (d0 : l0 = dense, d1 : l1 = compressed)}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<16x32xi32, #a>) -> ()

// -----
// expected-error@+1 {{use of undeclared identifier 'd0'}}
#a = #sparse_tensor.encoding<{map = {l0, l1} (d0 = l0, d1 = l1) -> (d0 : dense, d1 : compressed)}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<16x32xi32, #a>) -> ()

// -----
// expected-error@+1 {{expected '='}}
#a = #sparse_tensor.encoding<{map = {l0, l1} (d0 = l0, d1 = l1) -> (l0 : dense, l1 : compressed)}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<16x32xi32, #a>) -> ()

// -----
// expected-error@+1 {{use of undeclared identifier}}
#a = #sparse_tensor.encoding<{map = {l0, l1} (d0 = l0, d1 = l1) -> (l0 = dense, l1 = compressed)}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<16x32xi32, #a>) -> ()

// -----
// expected-error@+1 {{use of undeclared identifier 'd0'}}
#a = #sparse_tensor.encoding<{map = {l0, l1} (d0 = l0, d1 = l1) -> (d0 = l0 : dense, d1 = l1 : compressed)}>
func.func private @tensor_dimtolvl_mismatch(%arg0: tensor<16x32xi32, #a>) -> ()

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
  map = (d0 : #sparse_tensor<slice(-1, ?, 1)>, d1 : #sparse_tensor<slice(?, 4, 2)>) -> (d0 : dense, d1 : compressed)// expected-error{{expect positive value or ? for slice offset/size/stride}}
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

// expected-error@+1{{expected all singleton lvlTypes stored in the same memory layout (SoA vs AoS).}}
#COO_SoA = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed(nonunique), d1 : singleton(soa, nonunique), d2 : singleton)
}>
func.func private @sparse_coo(tensor<?x?xf32, #COO_SoA>)

// -----

// expected-error@+1{{SoA is only applicable to singleton lvlTypes.}}
#COO_SoA = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique, soa), d1 : singleton(soa))
}>
func.func private @sparse_coo(tensor<?x?xf32, #COO_SoA>)

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

// -----

// expected-error@+1 {{failed to infer lvlToDim from dimToLvl}}
#BSR = #sparse_tensor.encoding<{
  map = ( i, j ) ->
  ( i floordiv 2 : dense,
    j floordiv 3 : compressed,
    i            : dense,
    j mod 3      : dense
  )
}>
func.func private @BSR(%arg0: tensor<?x?xf64, #BSR>) {
  return
}

// -----

// expected-error@+1 {{failed to infer lvlToDim from dimToLvl}}
#BSR = #sparse_tensor.encoding<{
  map = ( i, j ) ->
  ( i            : dense,
    j floordiv 3 : compressed,
    i floordiv 3 : dense,
    j mod 3      : dense
  )
}>
func.func private @BSR(%arg0: tensor<?x?xf64, #BSR>) {
  return
}

// -----

// expected-error@+1 {{failed to infer lvlToDim from dimToLvl}}
#BSR = #sparse_tensor.encoding<{
  map = ( i, j ) ->
  ( i floordiv -3 : dense,
    j floordiv -3 : compressed,
    i mod 3 : dense,
    j mod 3      : dense
  )
}>
func.func private @BSR(%arg0: tensor<?x?xf64, #BSR>) {
  return
}

// -----

// expected-error@+1 {{expected lvlToDim to be an inverse of dimToLvl}}
#BSR_explicit = #sparse_tensor.encoding<{
  map =
  {il, jl, ii, jj}
  ( i = il * 3 + ii,
    j = jl * 2 + jj
  ) ->
  ( il = i floordiv 2 : dense,
    jl = j floordiv 3 : compressed,
    ii = i mod 2      : dense,
    jj = j mod 3      : dense
  )
}>
func.func private @BSR_explicit(%arg0: tensor<?x?xf64, #BSR_explicit>) {
  return
}

// -----

// expected-error@+6 {{expected structured size to be >= 0}}
#NOutOfM = #sparse_tensor.encoding<{
  map = ( i, j, k ) ->
  ( i            : dense,
    k floordiv 4 : dense,
    j            : dense,
    k mod 4      : structured[-2, 4]
  )
}>
func.func private @NOutOfM(%arg0: tensor<?x?x?xf64, #NOutOfM>) {
  return
}

// -----

// expected-error@+6 {{expected n <= m in n_out_of_m}}
#NOutOfM = #sparse_tensor.encoding<{
  map = ( i, j, k ) ->
  ( i            : dense,
    k floordiv 4 : dense,
    j            : dense,
    k mod 4      : structured[5, 4]
  )
}>
func.func private @NOutOfM(%arg0: tensor<?x?x?xf64, #NOutOfM>) {
  return
}

// -----

// expected-error@+1 {{expected all dense lvlTypes before a n_out_of_m level}}
#NOutOfM = #sparse_tensor.encoding<{
  map = ( i, j, k ) ->
  ( i            : dense,
    k floordiv 4 : compressed,
    j            : dense,
    k mod 4      : structured[2, 4]
  )
}>
func.func private @NOutOfM(%arg0: tensor<?x?x?xf64, #NOutOfM>) {
  return
}

// -----

// expected-error@+1 {{expected n_out_of_m to be the last level type}}
#NOutOfM = #sparse_tensor.encoding<{
  map = ( i, j, k ) ->
  ( i            : dense,
    k floordiv 4 : structured[2, 4],
    j            : dense,
    k mod 4      : compressed
  )
}>
func.func private @NOutOfM(%arg0: tensor<?x?x?xf64, #NOutOfM>) {
  return
}

// -----

// expected-error@+1 {{expected 1xm block structure for n_out_of_m level}}
#NOutOfM = #sparse_tensor.encoding<{
  map = ( i, j, k ) ->
  ( i            : dense,
    k floordiv 2 : dense,
    j            : dense,
    k mod 4      : structured[2, 4]
  )
}>
func.func private @NOutOfM(%arg0: tensor<?x?x?xf64, #NOutOfM>) {
  return
}

// -----

// expected-error@+1 {{expected coeffiencts of Affine expressions to be equal to m of n_out_of_m level}}
#NOutOfM = #sparse_tensor.encoding<{
  map = ( i, j, k ) ->
  ( i            : dense,
    k floordiv 2 : dense,
    j            : dense,
    k mod 2      : structured[2, 4]
  )
}>
func.func private @NOutOfM(%arg0: tensor<?x?x?xf64, #NOutOfM>) {
  return
}

// -----

// expected-error@+1 {{expected only one blocked level with the same coefficients}}
#NOutOfM = #sparse_tensor.encoding<{
  map = ( i, j, k ) ->
  ( i floordiv 2 : dense,
    i mod 2      : dense,
    j            : dense,
    k floordiv 4 : dense,
    k mod 4      : structured[2, 4]
  )
}>
func.func private @NOutOfM(%arg0: tensor<?x?x?xf64, #NOutOfM>) {
  return
}

// -----

#CSR_ExpType = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32,
  explicitVal = 1 : i32,
  implicitVal = 0.0 : f32
}>

// expected-error@+1 {{explicit value type mismatch between encoding and tensor element type: 'i32' != 'f32'}}
func.func private @sparse_csr(tensor<?x?xf32, #CSR_ExpType>)

// -----

#CSR_ImpType = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32,
  explicitVal = 1 : i32,
  implicitVal = 0.0 : f32
}>

// expected-error@+1 {{implicit value type mismatch between encoding and tensor element type: 'f32' != 'i32'}}
func.func private @sparse_csr(tensor<?x?xi32, #CSR_ImpType>)

// -----

// expected-error@+1 {{expected a numeric value for explicitVal}}
#CSR_ExpType = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32,
  explicitVal = "str"
}>
func.func private @sparse_csr(tensor<?x?xi32, #CSR_ExpType>)

// -----

// expected-error@+1 {{expected a numeric value for implicitVal}}
#CSR_ImpType = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32,
  implicitVal = "str"
}>
func.func private @sparse_csr(tensor<?x?xi32, #CSR_ImpType>)

// -----

#CSR_ImpVal = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32,
  implicitVal = 1 : i32
}>

// expected-error@+1 {{implicit value must be zero}}
func.func private @sparse_csr(tensor<?x?xi32, #CSR_ImpVal>)

// -----

#CSR_ImpVal = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32,
  implicitVal = 1.0 : f32
}>

// expected-error@+1 {{implicit value must be zero}}
func.func private @sparse_csr(tensor<?x?xf32, #CSR_ImpVal>)

// -----

#CSR_OnlyOnes = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 64,
  crdWidth = 64,
  explicitVal = #complex.number<:f32 1.0, 0.0>,
  implicitVal = #complex.number<:f32 1.0, 0.0>
}>

// expected-error@+1 {{implicit value must be zero}}
func.func private @sparse_csr(tensor<?x?xcomplex<f32>, #CSR_OnlyOnes>)
