// RUN: mlir-opt -canonicalize -split-input-file -verify-diagnostics %s | mlir-opt | FileCheck %s

// This tests that the `arm_sme.materialize_ssa_tile` placeholder is removed
// once it becomes unused, after lowering to control flow.

// -----

// CHECK-LABEL: @unused_materialize_ssa_tile_is_removed_from_blocks
// CHECK-NOT: arm_sme.materialize_ssa_tile
// CHECK-NOT: vector<[4]x[4]xf32>
func.func @unused_materialize_ssa_tile_is_removed_from_blocks(%arg0: memref<?x?xi32>) {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile = arm_sme.materialize_ssa_tile : vector<[4]x[4]xf32>
  cf.br ^bb1(%c0, %tile : index, vector<[4]x[4]xf32>)
^bb1(%1: index, %2: vector<[4]x[4]xf32>):  // 2 preds: ^bb0, ^bb2
  %3 = arith.cmpi slt, %1, %c10 : index
  cf.cond_br %3, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %4 = arith.addi %1, %c1 : index
  cf.br ^bb1(%4, %tile : index, vector<[4]x[4]xf32>)
^bb3:  // pred: ^bb1
  return
}
