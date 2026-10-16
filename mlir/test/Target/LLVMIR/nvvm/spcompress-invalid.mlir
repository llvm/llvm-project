// RUN: mlir-translate --mlir-to-llvmir -verify-diagnostics -split-input-file %s

llvm.func @spcompress_invalid_data_size(%spdesc: i32, %data: vector<2xi32>) {
  // expected-error @below {{Operation attributes and provided input data size doesn't match}}
  %metadata, %compressed_data = nvvm.spcompress %spdesc, %data
      comp_factor = sp2to4 index_size = b2 elem_size = b8 rep_factor = x2
      : i32, vector<2xi32> -> i32, vector<2xi32>
  llvm.return
}

// -----

llvm.func @spcompress_invalid_compressed_data_size(%spdesc: i32, %data: vector<4xi32>) {
  // expected-error @below {{Operation attributes and provided output compressed data size doesn't match}}
  %metadata, %compressed_data = nvvm.spcompress %spdesc, %data
      comp_factor = sp2to4 index_size = b2 elem_size = b8 rep_factor = x2
      : i32, vector<4xi32> -> i32, vector<4xi32>
  llvm.return
}

// -----

llvm.func @spcompress_invalid_metadata_size(%spdesc: i32, %data: vector<4xi32>) {
  // expected-error @below {{Operation attributes and provided output metadata size doesn't match}}
  %metadata, %compressed_data = nvvm.spcompress %spdesc, %data
      comp_factor = sp2to4 index_size = b2 elem_size = b8 rep_factor = x2
      : i32, vector<4xi32> -> vector<2xi32>, vector<2xi32>
  llvm.return
}

// -----

llvm.func @spcompress_invalid_elem_size(%spdesc: i32, %data: vector<2xi32>) {
  %metadata, %compressed_data = nvvm.spcompress %spdesc, %data
      // expected-error @below {{expected string or keyword containing one of the following enum values for attribute 'elemSize' [b8, b16]}}
      comp_factor = sp2to4 index_size = b2 elem_size = b32 rep_factor = x1
      : i32, vector<2xi32> -> i32, i32
  llvm.return
}

// -----

llvm.func @spcompress_invalid_comp_factor(%spdesc: i32, %data: vector<2xi32>) {
  // expected-error @below {{attribute 'compFactor' failed to satisfy constraint: Structured sparsity factor whose value is one of {sp2to4}}}
  %metadata, %compressed_data = nvvm.spcompress %spdesc, %data
      comp_factor = sp1to2 index_size = b2 elem_size = b8 rep_factor = x1
      : i32, vector<2xi32> -> i32, i32
  llvm.return
}

// -----

llvm.func @spcompress_scalable_data(%spdesc: i32, %data: vector<[2]xi32>) {
  // expected-error @below {{operand #1 must be fixed-length vector of 32-bit signless integer values of length 2/4/8/16/32/64/128 of ranks 1, but got 'vector<[2]xi32>'}}
  %metadata, %compressed_data = nvvm.spcompress %spdesc, %data
      comp_factor = sp2to4 index_size = b2 elem_size = b8 rep_factor = x1
      : i32, vector<[2]xi32> -> i32, i32
  llvm.return
}

// -----

llvm.func @spcompress_rank2_data(%spdesc: i32, %arg: vector<2xi32>) {
  %data = builtin.unrealized_conversion_cast %arg : vector<2xi32> to vector<1x2xi32>
  // expected-error @below {{operand #1 must be fixed-length vector of 32-bit signless integer values of length 2/4/8/16/32/64/128 of ranks 1, but got 'vector<1x2xi32>'}}
  %metadata, %compressed_data = nvvm.spcompress %spdesc, %data
      comp_factor = sp2to4 index_size = b2 elem_size = b8 rep_factor = x1
      : i32, vector<1x2xi32> -> i32, i32
  llvm.return
}

// -----

llvm.func @spcompress_scalable_compressed_data(%spdesc: i32, %data: vector<4xi32>) {
  // expected-error @below {{result #1 must be 32-bit signless integer or fixed-length vector of 32-bit signless integer values of length 2/4/8/16/32/64 of ranks 1, but got 'vector<[2]xi32>'}}
  %metadata, %compressed_data = nvvm.spcompress %spdesc, %data
      comp_factor = sp2to4 index_size = b2 elem_size = b8 rep_factor = x2
      : i32, vector<4xi32> -> i32, vector<[2]xi32>
  llvm.return
}
