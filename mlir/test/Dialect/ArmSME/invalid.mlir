// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// arm_sme.cast_tile_to_vector
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_cast_tile_to_vector__bad_tile_id_bitwidth(%tile_id : i8) -> vector<[8]x[8]xi16> {
  // expected-error@+1 {{op failed to verify that `tile_id` has the same number of bits as elements in `vector`}}
  %0 = arm_sme.cast_tile_to_vector %tile_id : i8 to vector<[8]x[8]xi16>
  return %0 : vector<[8]x[8]xi16>
}

// -----

func.func @arm_sme_cast_tile_to_vector__bad_vector_type_rank_1(%tile_id : i8) -> vector<[16]xi8> {
  // expected-error@+1 {{op result #0 must be vector<[16]x[16]xi8> of 8-bit signless integer values or vector<[8]x[8]xi16> of 16-bit signless integer values or vector<[4]x[4]xi32> of 32-bit signless integer values or vector<[2]x[2]xi64> of 64-bit signless integer values or vector<[1]x[1]xi128> of 128-bit signless integer values or vector<[8]x[8]xf16> of 16-bit float values or vector<[8]x[8]xbf16> of bfloat16 type values or vector<[4]x[4]xf32> of 32-bit float values or vector<[2]x[2]xf64> of 64-bit float values, but got 'vector<[16]xi8>'}}
  %0 = arm_sme.cast_tile_to_vector %tile_id : i8 to vector<[16]xi8>
  return %0 : vector<[16]xi8>
}

// -----

func.func @arm_sme_cast_tile_to_vector__bad_vector_type_i4(%tile_id : i8) -> vector<[16]x[16]xi4> {
  // expected-error@+1 {{op result #0 must be vector<[16]x[16]xi8> of 8-bit signless integer values or vector<[8]x[8]xi16> of 16-bit signless integer values or vector<[4]x[4]xi32> of 32-bit signless integer values or vector<[2]x[2]xi64> of 64-bit signless integer values or vector<[1]x[1]xi128> of 128-bit signless integer values or vector<[8]x[8]xf16> of 16-bit float values or vector<[8]x[8]xbf16> of bfloat16 type values or vector<[4]x[4]xf32> of 32-bit float values or vector<[2]x[2]xf64> of 64-bit float values, but got 'vector<[16]x[16]xi4>'}}
  %0 = arm_sme.cast_tile_to_vector %tile_id : i8 to vector<[16]x[16]xi4>
  return %0 : vector<[16]x[16]xi4>
}

// -----

func.func @arm_sme_cast_tile_to_vector__bad_vector_type_non_scalable_dim_0(%tile_id : i8) -> vector<16x[16]xi8> {
  // expected-error@+1 {{op result #0 must be vector<[16]x[16]xi8> of 8-bit signless integer values or vector<[8]x[8]xi16> of 16-bit signless integer values or vector<[4]x[4]xi32> of 32-bit signless integer values or vector<[2]x[2]xi64> of 64-bit signless integer values or vector<[1]x[1]xi128> of 128-bit signless integer values or vector<[8]x[8]xf16> of 16-bit float values or vector<[8]x[8]xbf16> of bfloat16 type values or vector<[4]x[4]xf32> of 32-bit float values or vector<[2]x[2]xf64> of 64-bit float values, but got 'vector<16x[16]xi8>'}}
  %0 = arm_sme.cast_tile_to_vector %tile_id : i8 to vector<16x[16]xi8>
  return %0 : vector<16x[16]xi8>
}

// -----

func.func @arm_sme_cast_tile_to_vector__bad_vector_type_non_scalable_dim_1(%tile_id : i8) -> vector<[16]x16xi8> {
  // expected-error@+1 {{op result #0 must be vector<[16]x[16]xi8> of 8-bit signless integer values or vector<[8]x[8]xi16> of 16-bit signless integer values or vector<[4]x[4]xi32> of 32-bit signless integer values or vector<[2]x[2]xi64> of 64-bit signless integer values or vector<[1]x[1]xi128> of 128-bit signless integer values or vector<[8]x[8]xf16> of 16-bit float values or vector<[8]x[8]xbf16> of bfloat16 type values or vector<[4]x[4]xf32> of 32-bit float values or vector<[2]x[2]xf64> of 64-bit float values, but got 'vector<[16]x16xi8>'}}
  %0 = arm_sme.cast_tile_to_vector %tile_id : i8 to vector<[16]x16xi8>
  return %0 : vector<[16]x16xi8>
}

// -----

func.func @arm_sme_cast_tile_to_vector_bad_shape(%tile_id : i8) -> vector<[4]x[16]xi8> {
  // expected-error@+1 {{op result #0 must be vector<[16]x[16]xi8> of 8-bit signless integer values or vector<[8]x[8]xi16> of 16-bit signless integer values or vector<[4]x[4]xi32> of 32-bit signless integer values or vector<[2]x[2]xi64> of 64-bit signless integer values or vector<[1]x[1]xi128> of 128-bit signless integer values or vector<[8]x[8]xf16> of 16-bit float values or vector<[8]x[8]xbf16> of bfloat16 type values or vector<[4]x[4]xf32> of 32-bit float values or vector<[2]x[2]xf64> of 64-bit float values, but got 'vector<[4]x[16]xi8>'}}
  %0 = arm_sme.cast_tile_to_vector %tile_id : i8 to vector<[4]x[16]xi8>
  return %0 : vector<[4]x[16]xi8>
}

//===----------------------------------------------------------------------===//
// arm_sme.cast_vector_to_tile
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_cast_vector_to_tile__bad_tile_id_bitwidth(%vector : vector<[1]x[1]xi128>) -> i32 {
  // expected-error@+1 {{op failed to verify that `tile_id` has the same number of bits as elements in `vector`}}
  %0 = arm_sme.cast_vector_to_tile %vector : vector<[1]x[1]xi128> to i32
  return %0 : i32
}

// -----

func.func @arm_sme_cast_vector_to_tile__bad_rank_1d(%vector : vector<[16]xi8>) -> i8 {
  // expected-error@+1 {{op operand #0 must be vector<[16]x[16]xi8> of 8-bit signless integer values or vector<[8]x[8]xi16> of 16-bit signless integer values or vector<[4]x[4]xi32> of 32-bit signless integer values or vector<[2]x[2]xi64> of 64-bit signless integer values or vector<[1]x[1]xi128> of 128-bit signless integer values or vector<[8]x[8]xf16> of 16-bit float values or vector<[8]x[8]xbf16> of bfloat16 type values or vector<[4]x[4]xf32> of 32-bit float values or vector<[2]x[2]xf64> of 64-bit float values, but got 'vector<[16]xi8>'}}
  %0 = arm_sme.cast_vector_to_tile %vector : vector<[16]xi8> to i8
  return %0 : i8
}

//===----------------------------------------------------------------------===//
// arm_sme.get_tile_id
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_get_tile_id__bad_type() -> i1 {
  // expected-error@+1 {{op result #0 must be 8-bit signless integer or 16-bit signless integer or 32-bit signless integer or 64-bit signless integer or 128-bit signless integer}}
  %0 = arm_sme.get_tile_id : i1
  return %0 : i1
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
