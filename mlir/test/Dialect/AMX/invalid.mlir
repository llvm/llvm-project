// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func @rowheight() {
  // expected-error@+1 {{'amx.tile_zero' op bad row height: 17}}
  %0 = amx.tile_zero : vector<17x16xbf16>
}

// -----

func @colwidth() {
  // expected-error@+1 {{'amx.tile_zero' op bad column width: 65}}
  %0 = amx.tile_zero : vector<16x65xi8>
}

// -----

func @col4bytemultiple() {
  // expected-error@+1 {{'amx.tile_zero' op bad column width: 5}}
  %0 = amx.tile_zero : vector<16x5xi8>
}

// -----

func @memtilesize(%arg0: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  // expected-error@+1 {{'amx.tile_load' op bad column width: 68}}
  %1 = amx.tile_load %arg0[%0, %0] : memref<?x?xf32> into vector<16x17xf32>
}

// -----

func @memindexsize(%arg0: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  // expected-error@+1 {{'amx.tile_load' op requires 2 indices}}
  %1 = amx.tile_load %arg0[%0] : memref<?x?xf32> into vector<16x16xf32>
}

// -----

func @multsize() {
  %0 = amx.tile_zero : vector<8x8xbf16>
  %1 = amx.tile_zero : vector<8x8xbf16>
  %2 = amx.tile_zero : vector<4x4xf32>
  // expected-error@+1 {{'amx.tile_mulf' op bad mult shape: 4 x 4 x 4}}
  %3 = amx.tile_mulf %0, %1, %2 : vector<8x8xbf16>, vector<8x8xbf16>, vector<4x4xf32>
}
