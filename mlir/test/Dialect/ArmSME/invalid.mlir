// RUN: mlir-opt %s -split-input-file -verify-diagnostics

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

// -----

func.func @arm_sme_get_tile_id__bad_type() -> i1 {
  // expected-error@+1 {{op result #0 must be 8-bit signless integer or 16-bit signless integer or 32-bit signless integer or 64-bit signless integer or 128-bit signless integer}}
  %0 = arm_sme.get_tile_id : i1
  return %0 : i1
}
