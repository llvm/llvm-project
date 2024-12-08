// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @misspelt_op_div(%A : memref<16x8xf32>, %B: memref<16x8xf32>, %C: memref<16x8xf32>) {
  // expected-error@+3 {{expected ::mlir::linalg::ElemwiseFn to be one of: exp, log, abs, ceil, floor}}
  // expected-error@+2 {{failed to parse ElemwiseFnAttr parameter}}
  // expected-error@+1 {{custom op 'linalg.elemwise' expected 'func_type' attribute}}
  linalg.elemwise func_type=#linalg.elemwise_fn<dive> ins(%A, %B: memref<16x8xf32>, memref<16x8xf32>) outs(%C: memref<16x8xf32>)
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @fewer_indexing_map(%A : memref<16x8xf32>, %B: memref<16x8xf32>, %C: memref<16x8xf32>) {
  // expected-error@+1 {{'linalg.elemwise' op expected the number of indexing_map (2) to be equal to the number of input/output operands (3)}}
  linalg.elemwise func_type=#linalg.elemwise_fn<div> indexing_maps = [#map, #map] ins(%A, %B: memref<16x8xf32>, memref<16x8xf32>) outs(%C: memref<16x8xf32>)
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @incorrect_transpose_map(%A : memref<8x16xf32>, %B: memref<16x8xf32>, %C: memref<16x8xf32>) {
  // expected-error@+1 {{'linalg.elemwise' op inferred input/output operand #1 has shape's dimension #0 to be 8, but found 16}}
  linalg.elemwise func_type=#linalg.elemwise_fn<div> indexing_maps = [#map, #map, #map] ins(%A, %B: memref<8x16xf32>, memref<16x8xf32>) outs(%C: memref<16x8xf32>)
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @incorrect_result_rank(%A : memref<8x16xf32>, %B: memref<8x16xf32>, %C: memref<8xf32>) {
  // expected-error@+1 {{'linalg.elemwise' op expected indexing_map #0 to have 1 dim(s) to match the number of loops}}
  linalg.elemwise func_type=#linalg.elemwise_fn<div> indexing_maps = [#map, #map, #map1] ins(%A, %B: memref<8x16xf32>, memref<8x16xf32>) outs(%C: memref<8xf32>)
  return
}

// -----
