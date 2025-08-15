// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @rowheight() {
  // expected-error@+1 {{'amx.tile_zero' op bad row height: 17}}
  %0 = amx.tile_zero : !amx.tile<17x16xbf16>
}

// -----

func.func @colwidth() {
  // expected-error@+1 {{'amx.tile_zero' op bad column width: 65}}
  %0 = amx.tile_zero : !amx.tile<16x65xi8>
}

// -----

func.func @col4bytemultiple() {
  // expected-error@+1 {{'amx.tile_zero' op bad column width: 5}}
  %0 = amx.tile_zero : !amx.tile<16x5xi8>
}

// -----

func.func @memtilesize(%arg0: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  // expected-error@+1 {{'amx.tile_load' op bad column width: 68}}
  %1 = amx.tile_load %arg0[%0, %0] : memref<?x?xf32> into !amx.tile<16x17xf32>
}

// -----

func.func @memindexsize(%arg0: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  // expected-error@+1 {{'amx.tile_load' op requires 2 indices}}
  %1 = amx.tile_load %arg0[%0] : memref<?x?xf32> into !amx.tile<16x16xf32>
}

// -----

func.func @multsize() {
  %0 = amx.tile_zero : !amx.tile<8x8xbf16>
  %1 = amx.tile_zero : !amx.tile<8x8xbf16>
  %2 = amx.tile_zero : !amx.tile<4x4xf32>
  // expected-error@+1 {{'amx.tile_mulf' op bad mult shape: 4 x 4 x 4}}
  %3 = amx.tile_mulf %0, %1, %2 : !amx.tile<8x8xbf16>, !amx.tile<8x8xbf16>, !amx.tile<4x4xf32>
}
