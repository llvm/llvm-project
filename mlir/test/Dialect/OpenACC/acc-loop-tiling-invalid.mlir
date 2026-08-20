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
  } attributes {independent = [#acc.device_type<none>]}
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
  } attributes {collapse = [2], collapseDeviceType = [#acc.device_type<none>], independent = [#acc.device_type<none>]}
  return
}
