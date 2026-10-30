// RUN: mlir-translate --mlir-to-llvmir -verify-diagnostics -split-input-file %s

llvm.func @spcompress_elem_type_mismatch(%spdesc: i32, %data: vector<8xi8>) {
  // expected-error @below {{'nvvm.sparse.compress' op expects data and compressed data to have the same element type}}
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<8xi8> -> i32, vector<4xi16>
  llvm.return
}

// -----

llvm.func @spcompress_data_too_small(%spdesc: i32, %data: vector<4xi8>) {
  // expected-error @below {{'nvvm.sparse.compress' op expects data to occupy 2 * repFactor 32-bit registers, where repFactor is a power of two up to 64}}
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<4xi8> -> i32, vector<2xi8>
  llvm.return
}

// -----

llvm.func @spcompress_rep_factor_not_power_of_2(%spdesc: i32, %data: vector<24xi8>) {
  // expected-error @below {{'nvvm.sparse.compress' op expects data to occupy 2 * repFactor 32-bit registers, where repFactor is a power of two up to 64}}
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<24xi8> -> i32, vector<12xi8>
  llvm.return
}

// -----

llvm.func @spcompress_invalid_compressed_data_size(%spdesc: i32, %data: vector<8xi8>) {
  // expected-error @below {{'nvvm.sparse.compress' op expects compressed data to have half as many elements as data}}
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<8xi8> -> i32, vector<8xi8>
  llvm.return
}

// -----

llvm.func @spcompress_invalid_metadata_size(%spdesc: i32, %data: vector<16xi8>) {
  // expected-error @below {{'nvvm.sparse.compress' op expects metadata to occupy 1 32-bit register(s)}}
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<16xi8> -> vector<2xi32>, vector<8xi8>
  llvm.return
}

// -----

llvm.func @spcompress_invalid_idx_size(%spdesc: i32, %data: vector<8xi8>) {
  // expected-error @below {{'nvvm.sparse.compress' op attribute 'idxSize' failed to satisfy constraint: 32-bit signless integer attribute whose value is one of {2, 4}}}
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 3
      : i32, vector<8xi8> -> i32, vector<4xi8>
  llvm.return
}

// -----

llvm.func @spcompress_invalid_elem_type(%spdesc: i32, %data: vector<2xi32>) {
  // expected-error @below {{'nvvm.sparse.compress' op operand #1 must be fixed-length vector of 8-bit signless integer or 16-bit signless integer values of ranks 1, but got 'vector<2xi32>'}}
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<2xi32> -> i32, vector<4xi8>
  llvm.return
}

// -----

llvm.func @spcompress_scalable_data(%spdesc: i32, %data: vector<[8]xi8>) {
  // expected-error @below {{'nvvm.sparse.compress' op operand #1 must be fixed-length vector of 8-bit signless integer or 16-bit signless integer values of ranks 1, but got 'vector<[8]xi8>'}}
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<[8]xi8> -> i32, vector<4xi8>
  llvm.return
}

// -----

llvm.func @spcompress_rank2_data(%spdesc: i32, %arg: vector<8xi8>) {
  %data = builtin.unrealized_conversion_cast %arg : vector<8xi8> to vector<1x8xi8>
  // expected-error @below {{'nvvm.sparse.compress' op operand #1 must be fixed-length vector of 8-bit signless integer or 16-bit signless integer values of ranks 1, but got 'vector<1x8xi8>'}}
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<1x8xi8> -> i32, vector<4xi8>
  llvm.return
}

// -----

llvm.func @spcompress_scalable_compressed_data(%spdesc: i32, %data: vector<8xi8>) {
  // expected-error @below {{'nvvm.sparse.compress' op result #1 must be fixed-length vector of 8-bit signless integer or 16-bit signless integer values of ranks 1, but got 'vector<[4]xi8>'}}
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<8xi8> -> i32, vector<[4]xi8>
  llvm.return
}

// -----

llvm.func @spcompress_invalid_metadata_type(%spdesc: i32, %data: vector<8xi8>) {
  // expected-error @below {{'nvvm.sparse.compress' op result #0 must be 32-bit signless integer or fixed-length vector of 32-bit signless integer values of length 1/2/4/8/16/32 of ranks 1, but got 'i64'}}
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<8xi8> -> i64, vector<4xi8>
  llvm.return
}
