// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

func.func @illegaltype(i) // expected-error {{expected non-function type}}

// -----

func.func @illegaltype() {
  %0 = arith.constant dense<0> : <vector 4 x f32> : vector<4 x f32> // expected-error {{expected non-function type}}
}

// -----

func.func @nestedtensor(tensor<tensor<i8>>) -> () // expected-error {{invalid tensor element type}}

// -----

// expected-error@+1 {{invalid dimension}}
#large_dim = tensor<9223372036854775808xf32>

// -----

func.func @illegalmemrefelementtype(memref<?xtensor<i8>>) -> () // expected-error {{invalid memref element type}}

// -----

func.func @illegalunrankedmemrefelementtype(memref<*xtensor<i8>>) -> () // expected-error {{invalid memref element type}}

// -----
// Test no map in memref type.
func.func @memrefs(memref<2x4xi8, >) // expected-error {{expected list element}}

// -----
// Test non-existent map in memref type.
func.func @memrefs(memref<2x4xi8, #map7>) // expected-error {{undefined symbol alias id 'map7'}}

// -----
// Test unsupported memory space.
func.func @memrefs(memref<2x4xi8, i8>) // expected-error {{unsupported memory space Attribute}}

// -----
// Test non-existent map in map composition of memref type.
#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @memrefs(memref<2x4xi8, #map0, #map8>) // expected-error {{undefined symbol alias id 'map8'}}

// -----
// Test multiple memory space error.
#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @memrefs(memref<2x4xi8, #map0, 1, 2>) // expected-error {{multiple memory spaces specified in memref type}}

// -----
// Test affine map after memory space.
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @memrefs(memref<2x4xi8, #map0, 1, #map1>) // expected-error {{expected memory space to be last in memref type}}

// -----
// Test dimension mismatch between memref and layout map.
// The error must be emitted even for the trivial identity layout maps that are
// dropped in type creation.
#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @memrefs(memref<42xi8, #map0>) // expected-error {{memref layout mismatch between rank and affine map: 1 != 2}}

// -----

// expected-error @below {{expected '<' after 'strided'}}
func.func private @memref_unfinished_strided() -> memref<?x?xf32, strided>

// -----

// expected-error @below {{expected '['}}
func.func private @memref_unfinished_strided() -> memref<?x?xf32, strided<>>

// -----

// expected-error @below {{expected a 64-bit signed integer or '?'}}
func.func private @memref_unfinished_stride_list() -> memref<?x?xf32, strided<[>>

// -----

// expected-error @below {{expected 'offset' after comma}}
func.func private @memref_missing_offset() -> memref<?x?xf32, strided<[], >>

// -----

// expected-error @below {{expected ':' after 'offset'}}
func.func private @memref_missing_offset_colon() -> memref<?x?xf32, strided<[], offset>>

// -----

// expected-error @below {{expected a 64-bit signed integer or '?'}}
func.func private @memref_missing_offset_value() -> memref<?x?xf32, strided<[], offset: >>

// -----

// expected-error @below {{expected '>'}}
func.func private @memref_incorrect_strided_ending() -> memref<?x?xf32, strided<[], offset: 32)>

// -----

// expected-error @below {{expected the number of strides to match the rank}}
func.func private @memref_strided_rank_mismatch() -> memref<?x?xf32, strided<[1]>>

// -----

func.func @tensor_encoding_mismatch(%arg0: tensor<8xi32, "enc">) -> (tensor<8xi32>) { // expected-note {{prior use here}}
  return %arg0: tensor<8xi32> // expected-error {{use of value '%arg0' expects different type than prior uses: 'tensor<8xi32>' vs 'tensor<8xi32, "enc">'}}
}

// -----

func.func @illegaltype(i21312312323120) // expected-error {{invalid integer width}}

// -----

// Test no nested vector.
// expected-error@+1 {{failed to verify 'elementType': VectorElementTypeInterface instance}}
func.func @vectors(vector<1 x vector<1xi32>>, vector<2x4xf32>)

// -----

// expected-error @+1 {{vector types must have positive constant sizes but got 0}}
func.func @zero_vector_type() -> vector<0xi32>

// -----

// expected-error @+1 {{vector types must have positive constant sizes but got 1, 0}}
func.func @zero_in_vector_type() -> vector<1x0xi32>

// -----

// expected-error @+1 {{expected non-function type}}
func.func @negative_vector_size() -> vector<-1xi32>

// -----

// expected-error @+1 {{expected non-function type}}
func.func @negative_in_vector_size() -> vector<1x-1xi32>

// -----

// expected-error @+1 {{expected non-function type}}
func.func @negative_memref_size() -> memref<-1xi32>

// -----

// expected-error @+1 {{expected non-function type}}
func.func @negative_in_memref_size() -> memref<1x-1xi32>

// -----

// expected-error @+1 {{expected non-function type}}
func.func @negative_tensor_size() -> tensor<-1xi32>

// -----

// expected-error @+1 {{expected non-function type}}
func.func @negative_in_tensor_size() -> tensor<1x-1xi32>

// -----

// expected-error @+1 {{invalid element type for complex}}
func.func @bad_complex(complex<memref<2x4xi8>>)

// -----

// expected-error @+1 {{expected '<' in complex type}}
func.func @bad_complex(complex memref<2x4xi8>>)

// -----

// expected-error @+1 {{expected '>' in complex type}}
func.func @bad_complex(complex<i32)

// -----
