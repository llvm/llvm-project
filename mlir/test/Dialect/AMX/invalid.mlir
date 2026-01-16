// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @tile_row_height() {
  // expected-error@+1 {{'amx.tile_zero' op bad row height: 17}}
  %0 = amx.tile_zero : !amx.tile<17x16xbf16>
  return
}

// -----

func.func @tile_col_width() {
  // expected-error@+1 {{'amx.tile_zero' op bad column width: 65}}
  %0 = amx.tile_zero : !amx.tile<16x65xi8>
  return
}

// -----

func.func @tile_element_type() {
  // expected-error@+1 {{failed to verify 'elementType'}}
  %0 = amx.tile_zero : !amx.tile<8x8xi16>
  return
}

// -----

func.func @tile_rank() {
  // expected-error@+1 {{'amx.tile_zero' op result #0 must be tile of}}
  %0 = amx.tile_zero : !amx.tile<32xi8>
  return
}

// -----

func.func @tile_col_4_byte_multiple() {
  // expected-error@+1 {{'amx.tile_zero' op bad column width: 5}}
  %0 = amx.tile_zero : !amx.tile<16x5xi8>
  return
}

// -----

func.func @load_base_tile_size(%arg0: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  // expected-error@+1 {{'amx.tile_load' op bad column width: 68}}
  %1 = amx.tile_load %arg0[%0, %0] : memref<?x?xf32> into !amx.tile<16x17xf32>
  return
}

// -----

func.func @store_base_tile_size(%arg0: memref<?x?xf32>, %arg1: !amx.tile<16x17xf32>) {
  %0 = arith.constant 0 : index
  // expected-error@+1 {{'amx.tile_store' op bad column width: 68}}
  amx.tile_store %arg0[%0, %0], %arg1 : memref<?x?xf32>, !amx.tile<16x17xf32>
  return
}

// -----

func.func @load_base_index_size(%arg0: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  // expected-error@+1 {{'amx.tile_load' op requires 2 indices}}
  %1 = amx.tile_load %arg0[%0] : memref<?x?xf32> into !amx.tile<16x16xf32>
  return
}

// -----

func.func @store_base_index_size(%arg0: memref<?x?xf32>, %arg1: !amx.tile<16x16xf32>) {
  %0 = arith.constant 0 : index
  // expected-error@+1 {{'amx.tile_store' op requires 2 indices}}
  amx.tile_store %arg0[%0], %arg1 : memref<?x?xf32>, !amx.tile<16x16xf32>
  return
}

// -----

func.func @load_base_rank(%arg0: memref<?xf32>) {
  %0 = arith.constant 0 : index
  // expected-error@+1 {{'amx.tile_load' op requires at least 2D memref}}
  %1 = amx.tile_load %arg0[%0] : memref<?xf32> into !amx.tile<16x16xf32>
  return
}

// -----

func.func @store_base_rank(%arg0: memref<?xf32>, %arg1: !amx.tile<16x16xf32>) {
  %0 = arith.constant 0 : index
  // expected-error@+1 {{'amx.tile_store' op requires at least 2D memref}}
  amx.tile_store %arg0[%0], %arg1 : memref<?xf32>, !amx.tile<16x16xf32>
  return
}

// -----

func.func @load_base_non_unit_stride(%arg0: memref<?x?xf32, strided<[?, ?]>>) {
  %0 = arith.constant 0 : index
  // expected-error@+1 {{'amx.tile_load' op requires memref with unit innermost stride}}
  %1 = amx.tile_load %arg0[%0, %0]
    : memref<?x?xf32, strided<[?, ?]>> into !amx.tile<16x16xf32>
  return
}

// -----

func.func @store_base_non_unit_stride(%arg0: memref<?x?xf32, strided<[?, ?]>>,
    %arg1: !amx.tile<16x16xf32>) {
  %0 = arith.constant 0 : index
  // expected-error@+1 {{'amx.tile_store' op requires memref with unit innermost stride}}
  amx.tile_store %arg0[%0, %0], %arg1
    : memref<?x?xf32, strided<[?, ?]>>, !amx.tile<16x16xf32>
  return
}

// -----

func.func @mulf_shape() {
  %0 = amx.tile_zero : !amx.tile<8x8xbf16>
  %1 = amx.tile_zero : !amx.tile<8x8xbf16>
  %2 = amx.tile_zero : !amx.tile<4x4xf32>
  // expected-error@+1 {{'amx.tile_mulf' op bad mult shape: 4 x 4 x 4}}
  %3 = amx.tile_mulf %0, %1, %2 : !amx.tile<8x8xbf16>, !amx.tile<8x8xbf16>, !amx.tile<4x4xf32>
  return
}

// -----

func.func @mulf_type_combination() {
  %0 = amx.tile_zero : !amx.tile<8x8xbf16>
  %1 = amx.tile_zero : !amx.tile<4x8xf16>
  %2 = amx.tile_zero : !amx.tile<8x4xf32>
  // expected-error@+1 {{'amx.tile_mulf' op unsupported type combination}}
  %3 = amx.tile_mulf %0, %1, %2 : !amx.tile<8x8xbf16>, !amx.tile<4x8xf16>, !amx.tile<8x4xf32>
  return
}

// -----

func.func @muli_shape() {
  %0 = amx.tile_zero : !amx.tile<8x8xi8>
  %1 = amx.tile_zero : !amx.tile<8x8xi8>
  %2 = amx.tile_zero : !amx.tile<4x4xi32>
  // expected-error@+1 {{'amx.tile_muli' op bad mult shape: 4 x 4 x 2}}
  %3 = amx.tile_muli %0, %1, %2 : !amx.tile<8x8xi8>, !amx.tile<8x8xi8>, !amx.tile<4x4xi32>
  return
}

// -----

func.func @muli_type_combination() {
  %0 = amx.tile_zero : !amx.tile<8x16xi8>
  %1 = amx.tile_zero : !amx.tile<8x16xi32>
  %2 = amx.tile_zero : !amx.tile<2x2xi32>
  // expected-error@+1 {{'amx.tile_muli' op operand #1 must be tile of 8-bit signless integer values}}
  %3 = amx.tile_muli %0, %1, %2 : !amx.tile<8x16xi8>, !amx.tile<8x16xi32>, !amx.tile<2x2xi32>
  return
}
