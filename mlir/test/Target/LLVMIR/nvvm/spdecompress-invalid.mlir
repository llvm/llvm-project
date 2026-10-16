// RUN: mlir-translate --mlir-to-llvmir -verify-diagnostics -split-input-file %s

llvm.func @spdecompress_data_too_small(%metadata: i32, %compdata: i32) {
  // expected-error @below {{Invalid compFactor, element size, and repeat factor combination}}
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp1to2 index_size = b4 elem_size = b8 rep_factor = x1
      : i32, i32 -> i32
  llvm.return
}

// -----

llvm.func @spdecompress_data_too_large(%metadata: i32, %compdata: vector<16xi32>) {
  // expected-error @below {{Invalid compFactor, element size, and repeat factor combination}}
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp1to16 index_size = b4 elem_size = b16 rep_factor = x32
      : i32, vector<16xi32> -> vector<128xi32>
  llvm.return
}

// -----

llvm.func @spdecompress_invalid_comp_factor_elem_size(%metadata: i32, %compdata: vector<2xi32>) {
  // expected-error @below {{Invalid compFactor and element size combination}}
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp4to8 index_size = b4 elem_size = b16 rep_factor = x1
      : i32, vector<2xi32> -> vector<4xi32>
  llvm.return
}

// -----

llvm.func @spdecompress_invalid_index_size(%metadata: i32, %compdata: i32) {
  // expected-error @below {{Invalid index size and target combination, target should be at most 4 if index size is 2}}
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp1to8 index_size = b2 elem_size = b8 rep_factor = x1
      : i32, i32 -> vector<2xi32>
  llvm.return
}

// -----

llvm.func @spdecompress_invalid_data_size(%metadata: i32, %compdata: i32) {
  // expected-error @below {{Operation attributes and provided output data size doesn't match}}
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp2to4 index_size = b4 elem_size = b8 rep_factor = x2
      : i32, i32 -> i32
  llvm.return
}

// -----

llvm.func @spdecompress_invalid_compressed_data_size(%metadata: i32, %compdata: vector<2xi32>) {
  // expected-error @below {{Operation attributes and provided input compressed data size doesn't match}}
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp2to4 index_size = b4 elem_size = b8 rep_factor = x2
      : i32, vector<2xi32> -> vector<2xi32>
  llvm.return
}

// -----

llvm.func @spdecompress_invalid_metadata_size(%metadata: vector<2xi32>, %compdata: i32) {
  // expected-error @below {{Operation attributes and provided input metadata size doesn't match}}
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp2to4 index_size = b4 elem_size = b8 rep_factor = x2
      : vector<2xi32>, i32 -> vector<2xi32>
  llvm.return
}

// -----

llvm.func @spdecompress_invalid_comp_factor(%metadata: i32, %compdata: i32) {
  %res = nvvm.spdecompress %metadata, %compdata
      // expected-error @below {{expected string or keyword containing one of the following enum values for attribute 'compFactor' [sp1to2, sp1to4, sp1to8, sp1to16, sp2to4, sp2to8, sp2to16, sp4to8, sp4to16]}}
      comp_factor = sp3to4 index_size = b4 elem_size = b8 rep_factor = x1
      : i32, i32 -> vector<2xi32>
  llvm.return
}

// -----

llvm.func @spdecompress_scalable_compressed_data(%metadata: i32, %compdata: vector<[2]xi32>) {
  // expected-error @below {{operand #1 must be 32-bit signless integer or fixed-length vector of 32-bit signless integer values of length 2/4/8/16/32/64 of ranks 1, but got 'vector<[2]xi32>'}}
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp1to2 index_size = b2 elem_size = b8 rep_factor = x8
      : i32, vector<[2]xi32> -> vector<4xi32>
  llvm.return
}

// -----

llvm.func @spdecompress_rank2_metadata(%arg: vector<2xi32>, %compdata: i32) {
  %metadata = builtin.unrealized_conversion_cast %arg : vector<2xi32> to vector<1x2xi32>
  // expected-error @below {{operand #0 must be 32-bit signless integer or fixed-length vector of 32-bit signless integer values of length 2/4/8/16/32/64 of ranks 1, but got 'vector<1x2xi32>'}}
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp1to2 index_size = b2 elem_size = b8 rep_factor = x2
      : vector<1x2xi32>, i32 -> i32
  llvm.return
}

// -----

llvm.func @spdecompress_scalable_data(%metadata: i32, %compdata: i32) {
  // expected-error @below {{result #0 must be 32-bit signless integer or fixed-length vector of 32-bit signless integer values of length 2/4/8/16/32/64/128 of ranks 1, but got 'vector<[2]xi32>'}}
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp1to2 index_size = b2 elem_size = b8 rep_factor = x4
      : i32, i32 -> vector<[2]xi32>
  llvm.return
}
