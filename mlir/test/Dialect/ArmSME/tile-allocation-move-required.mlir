// RUN: not mlir-opt %s -convert-scf-to-cf -test-arm-sme-tile-allocation 2>&1 | FileCheck %s

// `arm_sme.load_tile_slice` has operands, so it can't be cloned for free: merging two
// independently-allocated tiles here is a genuine conflict that needs a move.
//
// NOTE: Checked with plain FileCheck, not -verify-diagnostics: which of %vecA/%vecB
// gets named in the note is order-dependent, so it isn't deterministic across runs.

// CHECK: error: 'arm_sme.copy_tile' op tile operand allocated to different SME virtial tile (move required)
// CHECK: note: tile operand is:
func.func @overlapping_branches_with_real_tile_values(%cond: i1, %src: memref<?x?xi8>, %mask: vector<[16]xi1>) {
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %vecA = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %c0 : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  %vecB = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %c1 : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  %ret = scf.if %cond -> vector<[16]x[16]xi8> {
    scf.yield %vecA : vector<[16]x[16]xi8>
  } else {
    scf.yield %vecB : vector<[16]x[16]xi8>
  }
  "test.some_use"(%ret) : (vector<[16]x[16]xi8>) -> ()
  return
}
