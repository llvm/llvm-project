// RUN: mlir-opt %s -convert-scf-to-cf -test-arm-sme-tile-allocation -verify-diagnostics

// Select between tileA and tileB. This is currently unsupported as it would
// require inserting (runtime) tile moves.
func.func @selecting_between_different_tiles_is_unsupported(%dest : memref<?x?xi32>, %tileA : vector<[4]x[4]xi32>, %tileB : vector<[4]x[4]xi32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{op failed to rectify tile operand with tile result (move required)}}
  %tile = scf.if %cond -> vector<[4]x[4]xi32> {
    scf.yield %tileA : vector<[4]x[4]xi32>
  } else {
    scf.yield %tileB : vector<[4]x[4]xi32>
  }
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xi32>, vector<[4]x[4]xi32>
  return
}
