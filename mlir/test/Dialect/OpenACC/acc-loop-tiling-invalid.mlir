// RUN: mlir-opt %s -acc-loop-tiling -split-input-file -verify-diagnostics

// Test that tile size type wider than IV type is rejected

func.func @tile_wider_than_iv(%arg0: memref<100xf32>) {
  %c0 = arith.constant 0 : i32
  %c100 = arith.constant 100 : i32
  %c1 = arith.constant 1 : i32
  %c4_i64 = arith.constant 4 : i64  // i64 tile size with i32 IV
  // expected-error @+1 {{not yet implemented: tile size type (i64) is wider than loop IV type (i32)}}
  acc.loop tile({%c4_i64 : i64}) control(%i : i32) = (%c0 : i32) to (%c100 : i32) step (%c1 : i32) {
    acc.yield
  } independent
  return
}

// -----

// The first tile size applies to the innermost loop, so the i64 size below
// tiles the i32 IV %j.

func.func @tile_wider_than_inner_iv(%arg0: memref<100x50xf32>) {
  %c0_i64 = arith.constant 0 : i64
  %c100_i64 = arith.constant 100 : i64
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c0_i32 = arith.constant 0 : i32
  %c50_i32 = arith.constant 50 : i32
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  // expected-error @+1 {{not yet implemented: tile size type (i64) is wider than loop IV type (i32)}}
  acc.loop tile({%c4_i64 : i64, %c8_i32 : i32}) control(%i : i64, %j : i32) = (%c0_i64, %c0_i32 : i64, i32) to (%c100_i64, %c50_i32 : i64, i32) step (%c1_i64, %c1_i32 : i64, i32) {
    acc.yield
  } independent
  return
}

// -----

// A tile clause combined with a collapse clause on the same loop is not
// supported: the two clauses describe conflicting loop-association counts. The
// pass must diagnose it rather than silently drop the collapse clause.

func.func @tile_with_collapse(%arg0: memref<100x50xf32>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  // expected-error @below {{not yet implemented: a tile clause combined with a collapse clause on the same loop}}
  acc.loop tile({%c4 : index, %c8 : index}) control(%i : index, %j : index) = (%c0, %c0 : index, index) to (%c100, %c50 : index, index) step (%c1, %c1 : index, index) {
    %val = arith.index_castui %i : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i, %j] : memref<100x50xf32>
    acc.yield
  } collapse([2]) collapseDeviceType([#acc.device_type<none>]) independent
  return
}
