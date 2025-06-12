// RUN: mlir-opt %s -split-input-file -verify-diagnostics
func.func @misspelt_op_div(%A : memref<16x8xf32>, %B: memref<16x8xf32>, %C: memref<16x8xf32>) {
  // expected-error@+3 {{expected ::mlir::linalg::ElementwiseKind to be one of: exp, log, abs, ceil, floor}}
  // expected-error@+2 {{failed to parse ElementwiseKindAttr parameter}}
  // expected-error@+1 {{custom op 'linalg.elementwise' expected operation 'kind' attribute}}
  linalg.elementwise kind=#linalg.elementwise_kind<dive> ins(%A, %B: memref<16x8xf32>, memref<16x8xf32>) outs(%C: memref<16x8xf32>)
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @missing_indexing_map(%A : memref<16x8xf32>, %B: memref<16x8xf32>, %C: memref<16x8xf32>) {
  // expected-error@+1 {{'linalg.elementwise' op expected the number of indexing_map (2) to be equal to the number of input/output operands (3)}}
  linalg.elementwise kind=#linalg.elementwise_kind<div> indexing_maps = [#map, #map] ins(%A, %B: memref<16x8xf32>, memref<16x8xf32>) outs(%C: memref<16x8xf32>)
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @identity_map_when_transpose_expected(%A : memref<8x16xf32>, %B: memref<16x8xf32>, %C: memref<16x8xf32>) {
  // expected-error@+1 {{'linalg.elementwise' op inferred input/output operand #1 has shape's dimension #0 to be 8, but found 16}}
  linalg.elementwise kind=#linalg.elementwise_kind<div> indexing_maps = [#map, #map, #map] ins(%A, %B: memref<8x16xf32>, memref<16x8xf32>) outs(%C: memref<16x8xf32>)
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @incorrect_result_rank(%A : memref<8x16xf32>, %B: memref<8x16xf32>, %C: memref<8xf32>) {
  // expected-error@+1 {{'linalg.elementwise' op expected indexing_map #0 to have 1 dim(s) to match the number of loops}}
  linalg.elementwise kind=#linalg.elementwise_kind<div> indexing_maps = [#map, #map, #map1] ins(%A, %B: memref<8x16xf32>, memref<8x16xf32>) outs(%C: memref<8xf32>)
  return
}

// -----

func.func @unary_too_many_args(%A : memref<8x16x32xf32>, %B: memref<8x16x32xf32>, %C:  memref<8x16x32xf32>) {
  // expected-error@+3 {{custom op 'linalg.elementwise' [parseNamedStructuredOpRegion] ods-gen generated region expects 2 args, got 3}}
  // expected-error@+2 {{custom op 'linalg.elementwise' unable to parse elemwise op}}
  linalg.elementwise kind=#linalg.elementwise_kind<exp> ins(%A, %B : memref<8x16x32xf32>,  memref<8x16x32xf32>) outs(%C: memref<8x16x32xf32>)
  return
}

// -----

func.func @binary_too_few_args(%A : memref<8x16x32xf32>, %B: memref<8x16x32xf32>) {
  // expected-error@+3 {{custom op 'linalg.elementwise' [parseNamedStructuredOpRegion] ods-gen generated region expects 3 args, got 2}}
  // expected-error@+2 {{custom op 'linalg.elementwise' unable to parse elemwise op}}
  linalg.elementwise kind=#linalg.elementwise_kind<add> ins(%A : memref<8x16x32xf32>) outs(%B: memref<8x16x32xf32>)
  return
}
