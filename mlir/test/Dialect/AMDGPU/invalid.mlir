// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @mixing_packed_trunc_types(%arg0: f32, %arg1: vector<4xf8E5M2FNUZ>) -> vector<4xf8E4M3FNUZ> {
  // expected-error@+1 {{'amdgpu.packed_trunc_2xfp8' op existing values must have same type as result}}
  %ret = amdgpu.packed_trunc_2xfp8 %arg0, undef into %arg1[word 0] : f32 to vector<4xf8E4M3FNUZ> into vector<4xf8E5M2FNUZ>
  func.return %ret : vector<4xf8E4M3FNUZ>
}

// -----

func.func @mixing_packed_stoch_round_types(%arg0: f32, %arg1: i32, %arg2: vector<4xf8E5M2FNUZ>) -> vector<4xf8E4M3FNUZ> {
  // expected-error@+1 {{'amdgpu.packed_stoch_round_fp8' op existing values must have same type as result}}
  %ret = amdgpu.packed_stoch_round_fp8 %arg0 + %arg1 into %arg2[0] : f32 to vector<4xf8E4M3FNUZ> into vector<4xf8E5M2FNUZ>
  func.return %ret : vector<4xf8E4M3FNUZ>
}

// -----

func.func @bad_source_types(%a: vector<2xf32>, %b: vector<4xf16>,
                                %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op expected both non-small-float source operand types to match exactly}}
  %d = amdgpu.mfma 32x32x1 %a * %b + %c { blocks = 2 : i32, abid = 0 : i32, cbsz = 0 : i32} blgp = none : vector<2xf32>, vector<4xf16>, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @bad_source_types_f8(%a: vector<8xf8E5M2FNUZ>, %b: vector<8xi8>,
                                %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op expected both source operands to have small-float elements if one does}}
  %d = amdgpu.mfma 32x32x1 %a * %b + %c { blocks = 2 : i32, abid = 0 : i32, cbsz = 0 : i32} blgp = none : vector<8xf8E5M2FNUZ>, vector<8xi8>, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @bad_source_arguments(%a: vector<2xf32>, %b: vector<2xf32>,
                                %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op expected 1 source values for this operation but got 2}}
  %d = amdgpu.mfma 32x32x1 %a * %b + %c { blocks = 2 : i32, abid = 0 : i32, cbsz = 0 : i32} blgp = none : vector<2xf32>, vector<2xf32>, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @bad_source_arguments_i8(%a: vector<8xi8>, %b: vector<8xi8>,
                                   %c: vector<4xi32>) -> vector<4xi32> {
  // expected-error@+1 {{'amdgpu.mfma' op expected 4 source values for this operation but got 8}}
  %d = amdgpu.mfma 32x32x4 %a * %b + %c { blocks = 2 : i32, abid = 0 : i32, cbsz = 0 : i32} blgp = none : vector<8xi8>, vector<8xi8>, vector<4xi32>
  func.return %d : vector<4xi32>
}

// -----

func.func @bad_dest_type(%a: f32, %b: f32, %c: vector<16xf32>) -> vector<16xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op expected 32 result values for this operation but got 16}}
  %d = amdgpu.mfma 32x32x1 %a * %b + %c { blocks = 2 : i32, abid = 0 : i32, cbsz = 0 : i32} blgp = none : f32, f32, vector<16xf32>
  return %d : vector<16xf32>
}

// -----

func.func @f64_permuting_b(%a: f64, %b: f64, %c: vector<4xf64>) -> vector<4xf64> {
  // expected-error@+1 {{'amdgpu.mfma' op double-precision ops do not support permuting lanes of B}}
  %d = amdgpu.mfma 16x16x4 %a * %b + %c { abid = 0 : i32, cbsz = 0 : i32} blgp = bcast_first_32 : f64, f64, vector<4xf64>
  return %d : vector<4xf64>
}

// -----

func.func @f64_permuting_a(%a: f64, %b: f64, %c: vector<4xf64>) -> vector<4xf64> {
  // expected-error@+1 {{'amdgpu.mfma' op double-precision ops do not support permuting lanes of A}}
  %d = amdgpu.mfma 16x16x4 %a * %b + %c { abid = 0 : i32, cbsz = 1 : i32} blgp = none : f64, f64, vector<4xf64>
  return %d : vector<4xf64>
}

// -----

func.func @abid_without_bradcast(%a: f32, %b: f32, %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op block ID for permuting A (abid) must be below 2 ** cbsz}}
  %d = amdgpu.mfma 32x32x1 %a * %b + %c { blocks = 2 : i32, abid = 1 : i32, cbsz = 0 : i32} blgp = none : f32, f32, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @abid_too_large(%a: f32, %b: f32, %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op block ID for permuting A (abid) must be below 2 ** cbsz}}
  %d = amdgpu.mfma 32x32x1 %a * %b + %c { blocks = 2 : i32, abid = 2 : i32, cbsz = 1 : i32} blgp = none : f32, f32, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @no_negation(%a: f32, %b: f32, %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op negation flags only available for double-precision operations}}
  %d = amdgpu.mfma 32x32x1 %a * %b + %c { blocks = 2 : i32, abid = 0 : i32, cbsz = 0 : i32, negateA} blgp = none : f32, f32, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @mfma_invalid_m(%a: f32, %b: f32, %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op attribute 'm' failed to satisfy constraint: 32-bit signless integer attribute whose value is one of {4, 16, 32}}}
  %d = amdgpu.mfma 7x32x1 %a * %b + %c { abid = 0 : i32, cbsz = 0 : i32 } blgp = none : f32, f32, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @mfma_invalid_n(%a: f32, %b: f32, %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op attribute 'n' failed to satisfy constraint: 32-bit signless integer attribute whose value is one of {4, 16, 32}}}
  %d = amdgpu.mfma 32x7x1 %a * %b + %c { abid = 0 : i32, cbsz = 0 : i32 } blgp = none : f32, f32, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @mfma_invalid_k(%a: f32, %b: f32, %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op attribute 'k' failed to satisfy constraint: 32-bit signless integer attribute whose value is one of {1, 2, 4, 8, 16, 32, 64, 128}}}
  %d = amdgpu.mfma 32x32x3 %a * %b + %c { abid = 0 : i32, cbsz = 0 : i32 } blgp = none : f32, f32, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @mfma_invalid_blocks(%a: f32, %b: f32, %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op attribute 'blocks' failed to satisfy constraint: 32-bit signless integer attribute whose value is one of {1, 2, 4, 16}}}
  %d = amdgpu.mfma 32x32x1 %a * %b + %c { blocks = 7 : i32, abid = 0 : i32, cbsz = 0 : i32 } blgp = none : f32, f32, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @wmma_f16_i32(%arg0 : vector<16xf16>, %arg1 : vector<8xi32>) -> vector<8xi32> {
  // expected-error@+1 {{'amdgpu.wmma' op expected int sources with int destination}}
  %0 = amdgpu.wmma 16x16x16 %arg0 * %arg0 + %arg1 : vector<16xf16>, vector<16xf16>, vector<8xi32>
  func.return %0 : vector<8xi32>
}

// -----

func.func @wmma_i16_f32(%arg0 : vector<16xi8>, %arg1 : vector<8xf32>) -> vector<8xf32> {
  // expected-error@+1 {{'amdgpu.wmma' op expected float sources with float destination}}
  %0 = amdgpu.wmma 16x16x16 %arg0 * %arg0 + %arg1 : vector<16xi8>, vector<16xi8>, vector<8xf32>
  func.return %0 : vector<8xf32>
}

// -----

func.func @wmma_no_k_dim(%arg0 : vector<16xi8>, %arg1 : vector<8xi32>) -> vector<8xi32> {
  // expected-error@+1 {{'amdgpu.wmma' expected 3 dimensions in MNK dimension list}}
  %0 = amdgpu.wmma 16x16 %arg0 * %arg0 + %arg1 : vector<16xi8>, vector<16xi8>, vector<8xi32>
  func.return %0 : vector<8xi32>
}

// -----

func.func @wmma_wrong_n_dim(%arg0 : vector<16xi8>, %arg1 : vector<8xi32>) -> vector<8xi32> {
  // expected-error@+1 {{'amdgpu.wmma' op attribute 'n' failed to satisfy constraint: 32-bit signless integer attribute whose value is one of {16}}}
  %0 = amdgpu.wmma 16x32x16 %arg0 * %arg0 + %arg1 : vector<16xi8>, vector<16xi8>, vector<8xi32>
  func.return %0 : vector<8xi32>
}

// -----

func.func @wmma_wrong_k_dim(%arg0 : vector<16xi8>, %arg1 : vector<8xi32>) -> vector<8xi32> {
  // expected-error@+1 {{'amdgpu.wmma' op attribute 'k' failed to satisfy constraint: 32-bit signless integer attribute whose value is one of {4, 16, 32, 64, 128}}}
  %0 = amdgpu.wmma 16x16x24 %arg0 * %arg0 + %arg1 : vector<16xi8>, vector<16xi8>, vector<8xi32>
  func.return %0 : vector<8xi32>
}

// -----

func.func @wmma_source_length_mismatch(%arg0 : vector<8xf16>, %arg1 : vector<16xf16>, %arg2 : vector<8xf32>) -> vector<8xf32> {
  // expected-error@+1 {{'amdgpu.wmma' op source vectors have different lengths}}
  %0 = amdgpu.wmma 16x16x16 %arg0 * %arg1 + %arg2 : vector<8xf16>, vector<16xf16>, vector<8xf32>
  func.return %0 : vector<8xf32>
}

// -----

func.func @wmma_mismatched_float_types(%arg0 : vector<8xf16>, %arg1 : vector<8xbf16>, %arg2 : vector<8xf32>) -> vector<8xf32> {
  // expected-error@+1 {{'amdgpu.wmma' op source element types must match (except for fp8/bf8)}}
  %0 = amdgpu.wmma 16x16x16 %arg0 * %arg1 + %arg2 : vector<8xf16>, vector<8xbf16>, vector<8xf32>
  func.return %0 : vector<8xf32>
}

// -----

func.func @wmma_mismatched_int_types(%arg0 : vector<8xi8>, %arg1 : vector<8xi4>, %arg2 : vector<8xi32>) -> vector<8xi32> {
  // expected-error@+1 {{'amdgpu.wmma' op source element types must match (except for fp8/bf8)}}
  %0 = amdgpu.wmma 16x16x16 %arg0 * %arg1 + %arg2 : vector<8xi8>, vector<8xi4>, vector<8xi32>
  func.return %0 : vector<8xi32>
}

// -----

func.func @wmma_clamp_float(%arg0 : vector<8xf16>, %arg1 : vector<8xf32>) -> vector<8xf32> {
  // expected-error@+1 {{'amdgpu.wmma' op clamp flag is not supported for float types}}
  %0 = amdgpu.wmma 16x16x16 %arg0 * %arg0 + %arg1 {clamp} : vector<8xf16>, vector<8xf16>, vector<8xf32>
  func.return %0 : vector<8xf32>
}

// -----

func.func @wmma_unsignedA_float(%arg0 : vector<8xf16>, %arg1 : vector<8xf32>) -> vector<8xf32> {
  // expected-error@+1 {{'amdgpu.wmma' op unsigned flags are not supported for float types}}
  %0 = amdgpu.wmma 16x16x16 %arg0 * %arg0 + %arg1 {unsignedA} : vector<8xf16>, vector<8xf16>, vector<8xf32>
  func.return %0 : vector<8xf32>
}

// -----

func.func @wmma_unsignedB_float(%arg0 : vector<8xf16>, %arg1 : vector<8xf32>) -> vector<8xf32> {
  // expected-error@+1 {{'amdgpu.wmma' op unsigned flags are not supported for float types}}
  %0 = amdgpu.wmma 16x16x16 %arg0 * %arg0 + %arg1 {unsignedB} : vector<8xf16>, vector<8xf16>, vector<8xf32>
  func.return %0 : vector<8xf32>
}

// -----

// Missing `resetOffset`
func.func @fat_raw_buffer_cast_stripped_offset(%m: memref<8xi32, strided<[1], offset: ?>, #gpu.address_space<global>>) -> memref<8xi32, #amdgpu.address_space<fat_raw_buffer>> {
  // expected-error@+1 {{'amdgpu.fat_raw_buffer_cast' op expected result type to be 'memref<8xi32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>' but got 'memref<8xi32, #amdgpu.address_space<fat_raw_buffer>>'}}
  %ret = amdgpu.fat_raw_buffer_cast %m : memref<8xi32, strided<[1], offset: ?>, #gpu.address_space<global>> to memref<8xi32, #amdgpu.address_space<fat_raw_buffer>>
  func.return %ret : memref<8xi32, #amdgpu.address_space<fat_raw_buffer>>
}

// -----

func.func @fat_raw_buffer_cast_wrong_as(%m: memref<8xi32>) -> memref<8xi32, #amdgpu.address_space<buffer_rsrc>> {
  // expected-error@+1 {{'amdgpu.fat_raw_buffer_cast' op expected result type to be 'memref<8xi32, #amdgpu.address_space<fat_raw_buffer>>' but got 'memref<8xi32, #amdgpu.address_space<buffer_rsrc>>'}}
  %ret = amdgpu.fat_raw_buffer_cast %m : memref<8xi32> to memref<8xi32, #amdgpu.address_space<buffer_rsrc>>
  return %ret : memref<8xi32, #amdgpu.address_space<buffer_rsrc>>
}

// -----

func.func @fat_raw_buffer_cast_stripping_offset_affine_map(%m: memref<8xi32, affine_map<(d0)[s0] -> (d0 + s0)>>) -> memref<8xi32, #amdgpu.address_space<fat_raw_buffer>> {
  // expected-error@+1 {{'amdgpu.fat_raw_buffer_cast' op source type 'memref<8xi32, affine_map<(d0)[s0] -> (d0 + s0)>>' can't have its offset reset}}
  %ret = amdgpu.fat_raw_buffer_cast %m resetOffset : memref<8xi32, affine_map<(d0)[s0] -> (d0 + s0)>> to memref<8xi32, #amdgpu.address_space<fat_raw_buffer>>
  func.return %ret : memref<8xi32, #amdgpu.address_space<fat_raw_buffer>>
}

// -----

func.func @swizzle_invalid_type(%arg0 : si32) -> si32 {
  // expected-error@+1 {{'amdgpu.swizzle_bitmode' op operand #0 must be Integer or Float or fixed-length vector of Integer or Float values of ranks 1}}
  %0 = amdgpu.swizzle_bitmode %arg0 1 2 4 : si32
  func.return %0 : si32
}

// -----

func.func @swizzle_scalable_vec(%arg0 : vector<[4]xf32>) -> vector<[4]xf32> {
  // expected-error@+1 {{'amdgpu.swizzle_bitmode' op operand #0 must be Integer or Float or fixed-length vector of Integer or Float values of ranks 1}}
  %0 = amdgpu.swizzle_bitmode %arg0 1 2 4 : vector<[4]xf32>
  func.return %0 : vector<[4]xf32>
}

// -----

func.func @transpose_load_addrspace(%idx1 : index, %idx2 : index, %mem : memref<128x32xf16, 1>) -> vector<4xf16> {
  // expected-error@+1 {{'amdgpu.transpose_load' op source memory address space must be Workgroup}}
  %0 = amdgpu.transpose_load %mem[%idx1, %idx2] : memref<128x32xf16, 1> -> vector<4xf16>
  func.return %0 : vector<4xf16>
}

// -----

func.func @transpose_load_addrspace(%idx1 : index, %idx2 : index, %mem : memref<128x32xf16, 1>) -> vector<4xf16> {
  // expected-error@+1 {{'amdgpu.transpose_load' op source memory address space must be Workgroup}}
  %0 = amdgpu.transpose_load %mem[%idx1, %idx2] : memref<128x32xf16, 1> -> vector<4xf16>
  func.return %0 : vector<4xf16>
}

// -----

func.func @transpose_load_elem_f32(%idx1 : index, %idx2 : index, %mem : memref<128x32xf32, 3>) -> vector<4xf32> {
  // expected-error@+1 {{'amdgpu.transpose_load' op Unsupported element type size for transpose load: 32 bits}}
  %0 = amdgpu.transpose_load %mem[%idx1, %idx2] : memref<128x32xf32, 3> -> vector<4xf32>
  func.return %0 : vector<4xf32>
}

// -----

func.func @transpose_load_vector_size_f16(%idx1 : index, %idx2 : index, %mem : memref<128x32xf16, 3>) -> vector<2xf16> {
  // expected-error@+1 {{'amdgpu.transpose_load' op Transferring type size mismatch: expected num of elements: 4}}
  %0 = amdgpu.transpose_load %mem[%idx1, %idx2] : memref<128x32xf16, 3> -> vector<2xf16>
  func.return %0 : vector<2xf16>
}

// -----

func.func @transpose_load_vector_size_i4(%idx1 : index, %idx2 : index, %mem : memref<128x32xi4, 3>) -> vector<20xi4> {
  // expected-error@+1 {{'amdgpu.transpose_load' op Transferring type size mismatch: expected num of elements: 16}}
  %0 = amdgpu.transpose_load %mem[%idx1, %idx2] : memref<128x32xi4, 3> -> vector<20xi4>
  func.return %0 : vector<20xi4>
}

// -----

func.func @transpose_load_vector_size_i8(%idx1 : index, %idx2 : index, %mem : memref<128x32xi8, 3>) -> vector<20xi8> {
  // expected-error@+1 {{'amdgpu.transpose_load' op Transferring type size mismatch: expected num of elements: 8}}
  %0 = amdgpu.transpose_load %mem[%idx1, %idx2] : memref<128x32xi8, 3> -> vector<20xi8>
  func.return %0 : vector<20xi8>
}

// -----

func.func @transpose_load_vector_size_i8(%idx1 : index, %idx2 : index, %mem : memref<128x32xi6, 3>) -> vector<8xi6> {
  // expected-error@+1 {{'amdgpu.transpose_load' op Transferring type size mismatch: expected num of elements: 16}}
  %0 = amdgpu.transpose_load %mem[%idx1, %idx2] : memref<128x32xi6, 3> -> vector<8xi6>
  func.return %0 : vector<8xi6>
}

// -----

func.func @gather_to_lds_non_lds(%idx1 : index, %mem1 : memref<32xf16>, %mem2 : memref<32xf16>) {
  // expected-error@+1 {{'amdgpu.gather_to_lds' op destination memory address space must be Workgroup}}
  amdgpu.gather_to_lds %mem1[%idx1], %mem2[%idx1] : vector<2xf16>, memref<32xf16>, memref<32xf16>
  func.return
}

// -----

func.func @gather_to_lds_non_lds(%idx1 : index, %mem1 : memref<32xf16>, %mem2 : memref<32xf16, strided<[?]>, #gpu.address_space<workgroup>>) {
  // expected-error@+1 {{'amdgpu.gather_to_lds' op destination type inner most dim must be contiguous}}
  amdgpu.gather_to_lds %mem1[%idx1], %mem2[%idx1] : vector<2xf16>, memref<32xf16>, memref<32xf16, strided<[?]>, #gpu.address_space<workgroup>>
  func.return
}

// -----

func.func @scaled_mfma_invalid_m(%arg0 : vector<4xf8E8M0FNU>, %arg1 : vector<32xf4E2M1FN>, %arg2 : vector<16xf32>) -> vector<16xf32> {
  // expected-error@+1 {{'amdgpu.scaled_mfma' op attribute 'm' failed to satisfy constraint: 32-bit signless integer attribute whose value is one of {16, 32}}}
  %0 = amdgpu.scaled_mfma 8x32x64 (%arg0[0] * %arg1) * (%arg0[1] * %arg1) + %arg2 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<16xf32>
  func.return %0 : vector<16xf32>
}

// -----

func.func @scaled_mfma_invalid_n(%arg0 : vector<4xf8E8M0FNU>, %arg1 : vector<32xf4E2M1FN>, %arg2 : vector<16xf32>) -> vector<16xf32> {
  // expected-error@+1 {{'amdgpu.scaled_mfma' op attribute 'n' failed to satisfy constraint: 32-bit signless integer attribute whose value is one of {16, 32}}}
  %0 = amdgpu.scaled_mfma 32x8x64 (%arg0[0] * %arg1) * (%arg0[1] * %arg1) + %arg2 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<16xf32>
  func.return %0 : vector<16xf32>
}

// -----

func.func @scaled_mfma_invalid_k(%arg0 : vector<4xf8E8M0FNU>, %arg1 : vector<32xf4E2M1FN>, %arg2 : vector<16xf32>) -> vector<16xf32> {
  // expected-error@+1 {{'amdgpu.scaled_mfma' op attribute 'k' failed to satisfy constraint: 32-bit signless integer attribute whose value is one of {64, 128}}}
  %0 = amdgpu.scaled_mfma 32x32x32 (%arg0[0] * %arg1) * (%arg0[1] * %arg1) + %arg2 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<16xf32>
  func.return %0 : vector<16xf32>
}
