// RUN: mlir-translate --mlir-to-llvmir -verify-diagnostics -split-input-file %s

llvm.func @spdecompress_elem_type_mismatch(%metadata: i32, %compdata: vector<2xi8>) {
  // expected-error @below {{'nvvm.sparse.decompress' op expects data and compressed data to have the same element type}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 2 num_tgt = 4
      : i32, vector<2xi8> -> vector<4xi16>
  llvm.return
}

// -----

llvm.func @spdecompress_data_not_multiple_of_num_tgt(%metadata: i32, %compdata: vector<2xi8>) {
  // expected-error @below {{'nvvm.sparse.decompress' op expects data to have repFactor * num_tgt elements, where repFactor is a power of two up to 64}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 2 num_tgt = 4
      : i32, vector<2xi8> -> vector<6xi8>
  llvm.return
}

// -----

llvm.func @spdecompress_rep_factor_not_power_of_2(%metadata: i32, %compdata: vector<6xi8>) {
  // expected-error @below {{'nvvm.sparse.decompress' op expects data to have repFactor * num_tgt elements, where repFactor is a power of two up to 64}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 2 num_tgt = 4
      : i32, vector<6xi8> -> vector<12xi8>
  llvm.return
}

// -----

llvm.func @spdecompress_unsupported_num_src(%metadata: i32, %compdata: vector<3xi8>) {
  // expected-error @below {{'nvvm.sparse.decompress' op unsupported sparsity factor: compressed data has 3 elements for 1 group(s) of 4 elements}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 2 num_tgt = 4
      : i32, vector<3xi8> -> vector<4xi8>
  llvm.return
}

// -----

llvm.func @spdecompress_num_src_equals_num_tgt(%metadata: i32, %compdata: vector<4xi8>) {
  // expected-error @below {{'nvvm.sparse.decompress' op unsupported sparsity factor: compressed data has 4 elements for 1 group(s) of 4 elements}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 2 num_tgt = 4
      : i32, vector<4xi8> -> vector<4xi8>
  llvm.return
}

// -----

llvm.func @spdecompress_data_too_small(%metadata: i32, %compdata: vector<1xi8>) {
  // expected-error @below {{'nvvm.sparse.decompress' op expects data to be between 32 and 4096 bits}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 2 num_tgt = 2
      : i32, vector<1xi8> -> vector<2xi8>
  llvm.return
}

// -----

llvm.func @spdecompress_data_too_large(%metadata: vector<4xi32>, %compdata: vector<32xi16>) {
  // expected-error @below {{'nvvm.sparse.decompress' op expects data to be between 32 and 4096 bits}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 4 num_tgt = 16
      : vector<4xi32>, vector<32xi16> -> vector<512xi16>
  llvm.return
}

// -----

llvm.func @spdecompress_compressed_group_too_large(%metadata: i32, %compdata: vector<4xi16>) {
  // expected-error @below {{'nvvm.sparse.decompress' op expects at most 32 bits of compressed data per group}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 4 num_tgt = 8
      : i32, vector<4xi16> -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @spdecompress_invalid_idx_size_for_num_tgt(%metadata: i32, %compdata: vector<1xi8>) {
  // expected-error @below {{'nvvm.sparse.decompress' op expects num_tgt to be at most 4 if idx_size is 2}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 2 num_tgt = 8
      : i32, vector<1xi8> -> vector<8xi8>
  llvm.return
}

// -----

llvm.func @spdecompress_invalid_metadata_size(%metadata: vector<2xi32>, %compdata: vector<4xi8>) {
  // expected-error @below {{'nvvm.sparse.decompress' op expects metadata to occupy 1 32-bit register(s)}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 4 num_tgt = 4
      : vector<2xi32>, vector<4xi8> -> vector<8xi8>
  llvm.return
}

// -----

llvm.func @spdecompress_invalid_idx_size(%metadata: i32, %compdata: vector<2xi8>) {
  // expected-error @below {{'nvvm.sparse.decompress' op attribute 'idxSize' failed to satisfy constraint: 32-bit signless integer attribute whose value is one of {2, 4}}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 8 num_tgt = 4
      : i32, vector<2xi8> -> vector<4xi8>
  llvm.return
}

// -----

llvm.func @spdecompress_invalid_num_tgt(%metadata: i32, %compdata: vector<2xi8>) {
  // expected-error @below {{'nvvm.sparse.decompress' op attribute 'numTgt' failed to satisfy constraint: 32-bit signless integer attribute whose value is one of {2, 4, 8, 16}}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 2 num_tgt = 3
      : i32, vector<2xi8> -> vector<6xi8>
  llvm.return
}

// -----

llvm.func @spdecompress_scalable_compressed_data(%metadata: i32, %compdata: vector<[2]xi8>) {
  // expected-error @below {{'nvvm.sparse.decompress' op operand #1 must be fixed-length vector of 8-bit signless integer or 16-bit signless integer values of ranks 1, but got 'vector<[2]xi8>'}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 2 num_tgt = 4
      : i32, vector<[2]xi8> -> vector<4xi8>
  llvm.return
}

// -----

llvm.func @spdecompress_rank2_metadata(%arg: vector<2xi32>, %compdata: vector<2xi8>) {
  %metadata = builtin.unrealized_conversion_cast %arg : vector<2xi32> to vector<1x2xi32>
  // expected-error @below {{'nvvm.sparse.decompress' op operand #0 must be 32-bit signless integer or fixed-length vector of 32-bit signless integer values of length 1/2/4/8/16/32 of ranks 1, but got 'vector<1x2xi32>'}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 2 num_tgt = 4
      : vector<1x2xi32>, vector<2xi8> -> vector<4xi8>
  llvm.return
}

// -----

llvm.func @spdecompress_scalable_data(%metadata: i32, %compdata: vector<2xi8>) {
  // expected-error @below {{'nvvm.sparse.decompress' op result #0 must be fixed-length vector of 8-bit signless integer or 16-bit signless integer values of ranks 1, but got 'vector<[4]xi8>'}}
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 2 num_tgt = 4
      : i32, vector<2xi8> -> vector<[4]xi8>
  llvm.return
}
