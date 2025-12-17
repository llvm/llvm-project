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
