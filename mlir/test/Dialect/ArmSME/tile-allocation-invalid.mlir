// RUN: mlir-opt %s -allocate-arm-sme-tiles -split-input-file -verify-diagnostics

// -----

func.func @selecting_between_different_tiles_is_unsupported(%dest : memref<?x?xi32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %tileA = arm_sme.get_tile : vector<[4]x[4]xi32>
  %tileB = arm_sme.get_tile : vector<[4]x[4]xi32>
  // Select between tileA and tileB. This is currently unsupported as it would
  // require inserting tile move operations during tile allocation.
  %tile = scf.if %cond -> vector<[4]x[4]xi32> {
    scf.yield %tileA : vector<[4]x[4]xi32>
  } else {
    scf.yield %tileB : vector<[4]x[4]xi32>
  }
  // expected-error@+1 {{op already assigned different SME virtual tile!}}
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xi32>, vector<[4]x[4]xi32>
  return
}
