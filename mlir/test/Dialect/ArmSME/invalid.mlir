// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// arm_sme.get_tile
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_get_tile__bad_vector_type_rank_1() -> vector<[16]xi8> {
  // expected-error@+1 {{op result #0 must be a vector type that fits into a SME tile, but got 'vector<[16]xi8>'}}
  %0 = arm_sme.get_tile : vector<[16]xi8>
  return %0 : vector<[16]xi8>
}

// -----

func.func @arm_sme_get_tile__bad_vector_type_i4() -> vector<[16]x[16]xi4> {
  // expected-error@+1 {{op result #0 must be a vector type that fits into a SME tile, but got 'vector<[16]x[16]xi4>'}}
  %0 = arm_sme.get_tile : vector<[16]x[16]xi4>
  return %0 : vector<[16]x[16]xi4>
}

// -----

func.func @arm_sme_get_tile__bad_vector_type_non_scalable_dim_0() -> vector<16x[16]xi8> {
  // expected-error@+1 {{op result #0 must be a vector type that fits into a SME tile, but got 'vector<16x[16]xi8>'}}
  %0 = arm_sme.get_tile : vector<16x[16]xi8>
  return %0 : vector<16x[16]xi8>
}

// -----

func.func @arm_sme_get_tile__bad_vector_type_non_scalable_dim_1() -> vector<[16]x16xi8> {
  // expected-error@+1 {{op result #0 must be a vector type that fits into a SME tile, but got 'vector<[16]x16xi8>'}}
  %0 = arm_sme.get_tile : vector<[16]x16xi8>
  return %0 : vector<[16]x16xi8>
}

// -----

func.func @arm_sme_get_tile__bad_shape(%tile_id : i8) -> vector<[4]x[16]xi8> {
  // expected-error@+1 {{op result #0 must be a vector type that fits into a SME tile, but got 'vector<[4]x[16]xi8>'}}
  %0 = arm_sme.get_tile : vector<[4]x[16]xi8>
  return %0 : vector<[4]x[16]xi8>
}

//===----------------------------------------------------------------------===//
// arm_sme.move_vector_to_tile_slice
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_move_vector_to_tile_slice_i8__bad_vector_type(%vector : vector<[8]xi8>, %tile : vector<[16]x[16]xi8>, %tile_slice_index : index) -> vector<[16]x[16]xi8> {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{op failed to verify that type of 'vector' matches type of 'tile' slice}}
  %0 = arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[8]xi8> into vector<[16]x[16]xi8>
  return %0 : vector<[16]x[16]xi8>
}

// -----

func.func @arm_sme_move_vector_to_tile_slice_f32__bad_vector_type(%vector : vector<[8]xf32>, %tile : vector<[4]x[4]xf32>, %tile_slice_index : index) -> vector<[4]x[4]xf32> {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{op failed to verify that type of 'vector' matches type of 'tile' slice}}
  %0 = arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[8]xf32> into vector<[4]x[4]xf32>
  return %0 : vector<[4]x[4]xf32>
}

//===----------------------------------------------------------------------===//
// arm_sme.move_tile_slice_to_vector
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_move_tile_slice_to_vector__bad_result_type(%tile : vector<[4]x[4]xf32>, %tile_slice_index : index) -> vector<[2]xf64> {
  // expected-error@+1 {{op failed to verify that type of 'result' matches type of 'tile' slice}}
  %0 = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[2]xf64> from vector<[4]x[4]xf32>
  return %0 : vector<[2]xf64>
}

//===----------------------------------------------------------------------===//
// arm_sme.tile_load
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_tile_load__bad_padding_type(%src : memref<?x?xf64>, %pad : f32, %mask : vector<[2]x[2]xi1>) {
  %c0 = arith.constant 0 : index
  // expected-note@-2 {{prior use here}}
  // expected-error@+1 {{use of value '%pad' expects different type than prior uses: 'f64' vs 'f32'}}
  %tile = arm_sme.tile_load %src[%c0, %c0], %pad, %mask : memref<?x?xf64>, vector<[2]x[2]xf64>
  return
}

// -----

func.func @arm_sme_tile_load__bad_mask_type(%src : memref<?x?xf64>, %pad : f64, %mask : vector<[4]x[4]xi1>) {
  %c0 = arith.constant 0 : index
  // expected-note@-2 {{prior use here}}
  // expected-error@+1 {{use of value '%mask' expects different type than prior uses: 'vector<[2]x[2]xi1>' vs 'vector<[4]x[4]xi1>}}
  %tile = arm_sme.tile_load %src[%c0, %c0], %pad, %mask : memref<?x?xf64>, vector<[2]x[2]xf64>
  return
}

// -----

func.func @arm_sme_tile_load__pad_but_no_mask(%src : memref<?x?xf64>, %pad : f64) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{op failed to verify that both `padding` and `mask` should be provided or neither}}
  %tile = arm_sme.tile_load %src[%c0, %c0], %pad, : memref<?x?xf64>, vector<[2]x[2]xf64>
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.load_tile_slice
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_load_tile_slice__bad_mask_type(%src : memref<?x?xi8>, %mask : vector<[2]xi1>, %tile : vector<[16]x[16]xi8>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{op failed to verify that `mask` has i1 element type and the shape is a slice of `result`}}
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xi8>, vector<[2]xi1>, vector<[16]x[16]xi8>
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.tile_store
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_tile_store__bad_mask_type(%tile : vector<[16]x[16]xi8>, %mask : vector<[1]x[1]xi1>, %dest : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  // expected-note@-2 {{prior use here}}
  // expected-error@+1 {{use of value '%mask' expects different type than prior uses: 'vector<[16]x[16]xi1>' vs 'vector<[1]x[1]xi1>}}
  arm_sme.tile_store %tile, %dest[%c0, %c0], %mask : memref<?x?xi8>, vector<[16]x[16]xi8>
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.store_tile_slice
//===----------------------------------------------------------------------===//


// -----

func.func @arm_sme_store_tile_slice__bad_mask_type(%tile : vector<[16]x[16]xi8>, %tile_slice_index : index, %mask : vector<[8]xi1>, %dest : memref<?x?xi8>) -> () {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{op failed to verify that `mask` has i1 element type and the shape is a slice of `tile`}}
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xi8>, vector<[8]xi1>, vector<[16]x[16]xi8>
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.outerproduct
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_outerproduct__bad_result_type(%vecA: vector<[2]xi16>, %vecB: vector<[2]xi16>) -> vector<[2]x[2]xi16>
{
  // expected-error@+1 {{op result #0 must be a vector type that fits into a SME tile, but got 'vector<[2]x[2]xi16>'}}
  %0 = arm_sme.outerproduct %vecA, %vecB : vector<[2]xi16>, vector<[2]xi16>
  return %0 : vector<[2]x[2]xi16>
}

// -----

func.func @arm_sme_outerproduct__bad_vector_type(%vecA: vector<[4]xf32>, %vecB: vector<[8]xf32>) -> vector<[4]x[4]xf32>
{
  // expected-error@+1 {{op failed to verify that all of {lhs, rhs} have same type}}
  %0 = arm_sme.outerproduct %vecA, %vecB : vector<[4]xf32>, vector<[8]xf32>
  return %0 : vector<[4]x[4]xf32>
}

//===----------------------------------------------------------------------===//
// arm_sme.fmopa_2way
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_fmopa_2way__bad_rhs_vector_type(%vecA: vector<[8]xf16>, %vecB: vector<[4]xf32>) -> vector<[4]x[4]xf32>
{
  // expected-error@+1 {{op failed to verify that all of {lhs, rhs} have same type}}
  %0 = arm_sme.fmopa_2way %vecA, %vecB : vector<[8]xf16>, vector<[4]xf32> into vector<[4]x[4]xf32>
  return %0 : vector<[4]x[4]xf32>
}

// -----

func.func @arm_sme_fmopa_2way__bad_lhs_mask_type(%vecA: vector<[8]xf16>, %vecB: vector<[8]xf16>, %maskA : vector<[4]xi1>, %maskB : vector<[8]xi1>) -> vector<[4]x[4]xf32>
{
  // expected-note@-2 {{prior use here}}
  // expected-error@+1 {{use of value '%maskA' expects different type than prior uses: 'vector<[8]xi1>' vs 'vector<[4]xi1>}}
  %0 = arm_sme.fmopa_2way %vecA, %vecB masks(%maskA, %maskB) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  return %0 : vector<[4]x[4]xf32>
}

// -----

func.func @arm_sme_fmopa_2way__bad_rhs_mask_type(%vecA: vector<[8]xf16>, %vecB: vector<[8]xf16>, %maskA : vector<[8]xi1>, %maskB : vector<[4]xi1>) -> vector<[4]x[4]xf32>
{
  // expected-note@-2 {{prior use here}}
  // expected-error@+1 {{use of value '%maskB' expects different type than prior uses: 'vector<[8]xi1>' vs 'vector<[4]xi1>}}
  %0 = arm_sme.fmopa_2way %vecA, %vecB masks(%maskA, %maskB) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  return %0 : vector<[4]x[4]xf32>
}

// -----

func.func @arm_sme_fmopa_2way__no_rhs_mask(%vecA: vector<[8]xf16>, %vecB: vector<[8]xf16>, %maskA : vector<[8]xi1>) -> vector<[4]x[4]xf32>
{
  // expected-error@+1 {{op failed to verify that both `lhsMask` and `rhsMask` should be provided or neither}}
  %0 = arm_sme.fmopa_2way %vecA, %vecB masks(%maskA,) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  return %0 : vector<[4]x[4]xf32>
}

// -----

func.func @arm_sme_fmopa_2way__bad_acc_type(%vecA: vector<[8]xf16>, %vecB: vector<[8]xf16>) -> vector<[4]x[4]xf32>
{
  %acc = arm_sme.zero : vector<[2]x[2]xi64>
  // expected-note@-1 {{prior use here}}
  // expected-error@+1 {{use of value '%acc' expects different type than prior uses: 'vector<[4]x[4]xf32>' vs 'vector<[2]x[2]xi64>'}}
  %0 = arm_sme.fmopa_2way %vecA, %vecB masks(%maskA, %maskB) acc(%acc) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  return %0 : vector<[4]x[4]xf32>
}
