// RUN: mlir-opt %s -test-arm-sme-tile-allocation=preprocess-only -split-input-file | FileCheck %s

// This file tests the inserting copies for the SME tile allocation. Copies are
// inserted at `cf.br` ops (the predecessors to block arguments). Conditional
// branches are split to prevent conflicts (see cond_br_with_backedge).

// CHECK-LABEL: func.func @simple_branch(
//  CHECK-SAME:   %[[TILE:.*]]: vector<[4]x[4]xf32>)
//   %[[COPY:.*]] = arm_sme.copy_tile %[[TILE]] : vector<[4]x[4]xf32>
//   cf.br ^bb1(%[[COPY]] : vector<[4]x[4]xf32>)
// ^bb1(%[[BLOCK_ARG:.*]]: vector<[4]x[4]xf32>):

func.func @simple_branch(%tile : vector<[4]x[4]xf32>) {
  cf.br ^bb1(%tile: vector<[4]x[4]xf32>)
^bb1(%blockArg: vector<[4]x[4]xf32>):
  return
}

// -----

// Note: The ^POINTLESS_SHIM_FOR_BB2 block is added as the cond_br splitting does
// not check if it needs to insert a copy or not (there is no harm in the empty
// block though -- it will fold away later).

// CHECK-LABEL: func.func @cond_branch(
//  CHECK-SAME:   %[[COND:.*]]: i1, %[[TILE:.*]]: vector<[4]x[4]xf32>
//       CHECK:   cf.cond_br %[[COND]], ^[[BB1_COPIES:[[:alnum:]]+]], ^[[POINTLESS_SHIM_FOR_BB2:[[:alnum:]]+]]
//       CHECK: ^[[POINTLESS_SHIM_FOR_BB2]]:
//       CHECK:   cf.br ^[[BB2:.*]]
//       CHECK: ^[[BB1_COPIES]]:
//       CHECK:   arm_sme.copy_tile %[[TILE]] : vector<[4]x[4]xf32>
//       CHECK:   cf.br ^[[BB1:.*]]
func.func @cond_branch(%cond: i1, %tile: vector<[4]x[4]xf32>) {
  cf.cond_br %cond, ^bb1(%tile: vector<[4]x[4]xf32>), ^bb2
^bb1(%blockArg: vector<[4]x[4]xf32>):
  return
^bb2:
  return
}

// -----

// Reduction of a real world example that shows why we must split conditional branches.

// CHECK-LABEL: @cond_branch_with_backedge(
//  CHECK-SAME:    %[[TILEA:[[:alnum:]]+]]: vector<[4]x[4]xf32>, %[[TILEB:[[:alnum:]]+]]: vector<[4]x[4]xf32>,
//  CHECK-SAME:    %[[TILEC:[[:alnum:]]+]]: vector<[4]x[4]xf32>, %[[TILED:[[:alnum:]]+]]: vector<[4]x[4]xf32>,
//       CHECK:   %[[BB1_COPY_0:.*]] = arm_sme.copy_tile %[[TILEA]] : vector<[4]x[4]xf32>
//       CHECK:   cf.br ^bb1(%{{[[:alnum:]]+}}, %[[BB1_COPY_0]]
//       CHECK: ^bb1(%[[CURRENT_INDEX:.*]]: index, %[[ITER_TILE:.*]]: vector<[4]x[4]xf32>):
//       CHECK:   %[[CONTINUE_LOOP:.*]] = arith.cmpi
//       CHECK:   cf.cond_br %[[CONTINUE_LOOP]], ^[[BB2_COPIES:[[:alnum:]]+]], ^[[BB3_COPIES:[[:alnum:]]+]]
//       CHECK: ^[[BB3_COPIES]]:
//  CHECK-NEXT:   %[[BB3_COPY_0:.*]] = arm_sme.copy_tile %[[ITER_TILE]] : vector<[4]x[4]xf32>
//  CHECK-NEXT:   %[[BB3_COPY_1:.*]] = arm_sme.copy_tile %[[TILEB]] : vector<[4]x[4]xf32>
//  CHECK-NEXT:   %[[BB3_COPY_2:.*]] = arm_sme.copy_tile %[[TILEC]] : vector<[4]x[4]xf32>
//  CHECK-NEXT:   %[[BB3_COPY_3:.*]] = arm_sme.copy_tile %[[TILED]] : vector<[4]x[4]xf32>
//  CHECK-NEXT:   cf.br ^[[BB3:[[:alnum:]]+]](%[[BB3_COPY_0]], %[[BB3_COPY_1]], %[[BB3_COPY_2]], %[[BB3_COPY_3]]
//       CHECK: ^[[BB2_COPIES]]:
//  CHECK-NEXT:   cf.br ^[[BB2:[[:alnum:]]+]]
//       CHECK: ^[[BB2]]:
//  CHECK-NEXT:   %[[NEXT_TILE:.*]] = arm_sme.move_vector_to_tile_slice %{{.*}}, %[[ITER_TILE]]
//       CHECK:   %[[BB1_COPY_1:.*]] = arm_sme.copy_tile %[[NEXT_TILE]] : vector<[4]x[4]xf32>
//       CHECK:   cf.br ^bb1(%{{[[:alnum:]]+}}, %[[BB1_COPY_1]]
//       CHECK: ^[[BB3]](%{{.*}}: vector<[4]x[4]xf32>):
//  CHECK-NEXT:   return
func.func @cond_branch_with_backedge(%tileA: vector<[4]x[4]xf32>, %tileB: vector<[4]x[4]xf32>, %tileC: vector<[4]x[4]xf32>, %tileD: vector<[4]x[4]xf32>, %slice: vector<[4]xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  // Live here: %tileA, %tileB, %tileC, %tileD
  cf.br ^bb1(%c0, %tileA : index, vector<[4]x[4]xf32>)
^bb1(%currentIndex: index, %iterTile: vector<[4]x[4]xf32>):
  %continueLoop = arith.cmpi slt, %currentIndex, %c10 : index
  // Live here: %iterTile, %tileB, %tileC, %tileD
  // %iterTile, %tileB, %tileC, %tileD are live out (in the ^bb2 case). If we
  // inserted the (four) `arm_sme.copy_tile` operations here we would run out of tiles.
  // However, note that the copies are only needed if we take the ^bb3 path. So, if we add
  // a new block along that path we can insert the copies without any conflicts.
  cf.cond_br %continueLoop, ^bb2, ^bb3(%iterTile, %tileB, %tileC, %tileD : vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>)
^bb2:
  // Live here: %iterTile, %tileB, %tileC, %tileD
  %nextTile = arm_sme.move_vector_to_tile_slice %slice, %iterTile, %currentIndex : vector<[4]xf32> into vector<[4]x[4]xf32>
  %nextIndex = arith.addi %currentIndex, %c1 : index
  cf.br ^bb1(%nextIndex, %nextTile : index, vector<[4]x[4]xf32>)
^bb3(%finalTileA: vector<[4]x[4]xf32>, %finalTileB: vector<[4]x[4]xf32>, %finalTileC: vector<[4]x[4]xf32>, %finalTileD: vector<[4]x[4]xf32>):
  // Live here: %finalTileA, %finalTileB, %finalTileC, %finalTileD
  return
}

// -----

// CHECK-LABEL: @tile_dominance
// CHECK-NOT: arm_sme.copy_tile
func.func @tile_dominance(%arg0: vector<[4]x[4]xf32>) {
  cf.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb4
  "test.some_use"(%arg0) : (vector<[4]x[4]xf32>) -> ()
  return
^bb2:  // no predecessors
  %0 = arm_sme.get_tile : vector<[4]x[4]xf32>
  cf.br ^bb3
^bb3:  // pred: ^bb2
  "test.some_use"(%0) : (vector<[4]x[4]xf32>) -> ()
  return
^bb4:  // no predecessors
  cf.br ^bb1
^bb5:  // no predecessors
  return
}

// -----

// CHECK-LABEL: func.func @cond_branch_true_and_false_tile_args(
//  CHECK-SAME:   %[[COND:.*]]: i1, %[[TILE:.*]]: vector<[4]x[4]xf32>
//  CHECK-NEXT:   cf.cond_br %[[COND]], ^[[BB1_COPIES:[[:alnum:]]+]], ^[[BB2_COPIES:[[:alnum:]]+]]
//       CHECK: ^[[BB2_COPIES]]:
//  CHECK-NEXT:   %[[COPY_0:.*]] = arm_sme.copy_tile %[[TILE]] :  vector<[4]x[4]xf32>
//  CHECK-NEXT:   cf.br ^[[BB2:[[:alnum:]]+]](%[[COPY_0]]
//       CHECK: ^[[BB1_COPIES]]:
//  CHECK-NEXT:   %[[COPY_1:.*]] = arm_sme.copy_tile %[[TILE]] :  vector<[4]x[4]xf32>
//  CHECK-NEXT:   cf.br ^[[BB1:[[:alnum:]]+]](%[[COPY_1]]
//       CHECK: ^[[BB1]]{{.*}}:
//  CHECK-NEXT:   return
//       CHECK: ^[[BB2]]{{.*}}:
//  CHECK-NEXT:   return
func.func @cond_branch_true_and_false_tile_args(%cond: i1, %tile: vector<[4]x[4]xf32>) {
  cf.cond_br %cond, ^bb1(%tile: vector<[4]x[4]xf32>), ^bb2(%tile: vector<[4]x[4]xf32>)
^bb1(%blockArg0: vector<[4]x[4]xf32>):
  return
^bb2(%blockArg1: vector<[4]x[4]xf32>):
  return
}

// -----

// CHECK-LABEL: @multiple_predecessors
//      CHECK: ^bb1:
// CHECK-NEXT:   %[[TILE:.*]] = arm_sme.get_tile : vector<[4]x[4]xf32>
// CHECK-NEXT:   %[[COPY_0:.*]] = arm_sme.copy_tile %[[TILE]] : vector<[4]x[4]xf32>
// CHECK-NEXT:    cf.br ^bb3(%[[COPY_0]] : vector<[4]x[4]xf32>)
//      CHECK: ^bb2:
// CHECK-NEXT:   %[[ZERO:.*]] = arm_sme.zero : vector<[4]x[4]xf32>
// CHECK-NEXT:   %[[COPY_1:.*]] = arm_sme.copy_tile %[[ZERO]] : vector<[4]x[4]xf32>
// CHECK-NEXT:    cf.br ^bb3(%[[COPY_1]] : vector<[4]x[4]xf32>)
//      CHECK: ^bb3({{.*}}):
// CHECK-NEXT:  return
func.func @multiple_predecessors(%cond: i1)
{
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  cf.br ^bb3(%tile : vector<[4]x[4]xf32>)
^bb2:
  %zero = arm_sme.zero : vector<[4]x[4]xf32>
  cf.br ^bb3(%zero : vector<[4]x[4]xf32>)
^bb3(%blockArg: vector<[4]x[4]xf32>): // pred: ^bb1, ^bb2
  return
}
