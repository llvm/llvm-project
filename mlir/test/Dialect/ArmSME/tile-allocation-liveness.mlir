// RUN: mlir-opt %s -convert-scf-to-cf -test-arm-sme-tile-allocation -split-input-file -verify-diagnostics | FileCheck %s
// RUN: mlir-opt %s -convert-scf-to-cf -test-arm-sme-tile-allocation=dump-tile-live-ranges -mlir-disable-threading -split-input-file -verify-diagnostics 2>&1 >/dev/null | FileCheck %s --check-prefix=CHECK-LIVE-RANGE

// This file tests some simple aspects of using liveness in the SME tile allocator.
// Note: We use -convert-scf-to-cf first as the tile allocator expects CF, but
// some of these tests are written in SCF (to make things easier to follow).

//  CHECK-LIVE-RANGE-LABEL: @constant_with_multiple_users
//        CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//        CHECK-LIVE-RANGE: ^bb0:
//        CHECK-LIVE-RANGE: S  arm_sme.zero
//   CHECK-LIVE-RANGE-NEXT: |S arm_sme.move_vector_to_tile_slice
//   CHECK-LIVE-RANGE-NEXT: || arm_sme.move_vector_to_tile_slice
//   CHECK-LIVE-RANGE-NEXT: |E test.some_use
//   CHECK-LIVE-RANGE-NEXT: E  test.some_use

// CHECK-LABEL: @constant_with_multiple_users(
// CHECK-SAME:                                %[[VECTOR_A:.*]]: vector<[4]xf32>, %[[VECTOR_B:.*]]: vector<[4]xf32>
func.func @constant_with_multiple_users(%a: vector<[4]xf32>, %b: vector<[4]xf32>, %index: index) {
  // CHECK-NEXT: %[[ZERO_TILE_0:.*]] = arm_sme.zero {tile_id = 0 : i32} : vector<[4]x[4]xf32>
  // CHECK-NEXT: %[[ZERO_TILE_1:.*]] = arm_sme.zero {tile_id = 1 : i32} : vector<[4]x[4]xf32>
  // CHECK-NEXT: %[[INSERT_TILE_1:.*]] = arm_sme.move_vector_to_tile_slice %[[VECTOR_A]], %[[ZERO_TILE_1]], %{{.*}} {tile_id = 1 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
  // CHECK-NEXT: %[[INSERT_TILE_0:.*]] = arm_sme.move_vector_to_tile_slice %[[VECTOR_B]], %[[ZERO_TILE_0]], %{{.*}} {tile_id = 0 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
  %zero = arm_sme.zero : vector<[4]x[4]xf32>
  %tile_a = arm_sme.move_vector_to_tile_slice %a, %zero, %index : vector<[4]xf32> into vector<[4]x[4]xf32>
  %tile_b = arm_sme.move_vector_to_tile_slice %b, %zero, %index : vector<[4]xf32> into vector<[4]x[4]xf32>
  "test.some_use"(%tile_a) : (vector<[4]x[4]xf32>) -> ()
  "test.some_use"(%tile_b) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

//  CHECK-LIVE-RANGE-LABEL: @value_with_multiple_users
//        CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//        CHECK-LIVE-RANGE: ^bb0:
//   CHECK-LIVE-RANGE-NEXT: |S arm_sme.move_vector_to_tile_slice
//   CHECK-LIVE-RANGE-NEXT: || arm_sme.move_vector_to_tile_slice
//   CHECK-LIVE-RANGE-NEXT: |E test.some_use
//   CHECK-LIVE-RANGE-NEXT: E  test.some_use

// expected-note@below {{tile operand is: <block argument> of type 'vector<[4]x[4]xf32>'}}
func.func @value_with_multiple_users(%tile: vector<[4]x[4]xf32>, %a: vector<[4]xf32>, %b: vector<[4]xf32>, %index: index) {
  // expected-error@below {{op tile operand allocated to different SME virtial tile (move required)}}
  %tile_a = arm_sme.move_vector_to_tile_slice %a, %tile, %index : vector<[4]xf32> into vector<[4]x[4]xf32>
  %tile_b = arm_sme.move_vector_to_tile_slice %b, %tile, %index : vector<[4]xf32> into vector<[4]x[4]xf32>
  "test.some_use"(%tile_a) : (vector<[4]x[4]xf32>) -> ()
  "test.some_use"(%tile_b) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

//  CHECK-LIVE-RANGE-LABEL: @reuse_tiles_after_initial_use
//        CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//        CHECK-LIVE-RANGE: ^bb0:
//   CHECK-LIVE-RANGE-NEXT: S        arm_sme.get_tile
//   CHECK-LIVE-RANGE-NEXT: |S       arm_sme.get_tile
//   CHECK-LIVE-RANGE-NEXT: ||S      arm_sme.get_tile
//   CHECK-LIVE-RANGE-NEXT: |||S     arm_sme.get_tile
//   CHECK-LIVE-RANGE-NEXT: ||||     test.dummy
//   CHECK-LIVE-RANGE-NEXT: ||||     test.dummy
//   CHECK-LIVE-RANGE-NEXT: ||||     test.dummy
//   CHECK-LIVE-RANGE-NEXT: E|||     test.some_use
//   CHECK-LIVE-RANGE-NEXT:  E||     test.some_use
//   CHECK-LIVE-RANGE-NEXT:   E|     test.some_use
//   CHECK-LIVE-RANGE-NEXT:    E     test.some_use
//   CHECK-LIVE-RANGE-NEXT:     S    arm_sme.zero
//   CHECK-LIVE-RANGE-NEXT:     |S   arm_sme.zero
//   CHECK-LIVE-RANGE-NEXT:     ||S  arm_sme.zero
//   CHECK-LIVE-RANGE-NEXT:     |||S arm_sme.zero
//   CHECK-LIVE-RANGE-NEXT:     |||| test.dummy
//   CHECK-LIVE-RANGE-NEXT:     |||| test.dummy
//   CHECK-LIVE-RANGE-NEXT:     |||| test.dummy
//   CHECK-LIVE-RANGE-NEXT:     E||| test.some_use
//   CHECK-LIVE-RANGE-NEXT:      E|| test.some_use
//   CHECK-LIVE-RANGE-NEXT:       E| test.some_use
//   CHECK-LIVE-RANGE-NEXT:        E test.some_use

// CHECK-LABEL: @reuse_tiles_after_initial_use
func.func @reuse_tiles_after_initial_use() {
  // CHECK: arm_sme.get_tile {tile_id = 0 : i32}
  // CHECK: arm_sme.get_tile {tile_id = 1 : i32}
  // CHECK: arm_sme.get_tile {tile_id = 2 : i32}
  // CHECK: arm_sme.get_tile {tile_id = 3 : i32}
  %tile_a = arm_sme.get_tile : vector<[4]x[4]xf32>
  %tile_b = arm_sme.get_tile : vector<[4]x[4]xf32>
  %tile_c = arm_sme.get_tile : vector<[4]x[4]xf32>
  %tile_d = arm_sme.get_tile : vector<[4]x[4]xf32>
  "test.dummy"(): () -> ()
  "test.dummy"(): () -> ()
  "test.dummy"(): () -> ()
  "test.some_use"(%tile_a) : (vector<[4]x[4]xf32>) -> ()
  "test.some_use"(%tile_b) : (vector<[4]x[4]xf32>) -> ()
  "test.some_use"(%tile_c) : (vector<[4]x[4]xf32>) -> ()
  "test.some_use"(%tile_d) : (vector<[4]x[4]xf32>) -> ()
  // CHECK: arm_sme.zero {tile_id = 0 : i32}
  // CHECK: arm_sme.zero {tile_id = 1 : i32}
  // CHECK: arm_sme.zero {tile_id = 2 : i32}
  // CHECK: arm_sme.zero {tile_id = 3 : i32}
  %tile_1 = arm_sme.zero : vector<[4]x[4]xf32>
  %tile_2 = arm_sme.zero : vector<[4]x[4]xf32>
  %tile_3 = arm_sme.zero : vector<[4]x[4]xf32>
  %tile_4 = arm_sme.zero : vector<[4]x[4]xf32>
  "test.dummy"(): () -> ()
  "test.dummy"(): () -> ()
  "test.dummy"(): () -> ()
  "test.some_use"(%tile_1) : (vector<[4]x[4]xf32>) -> ()
  "test.some_use"(%tile_2) : (vector<[4]x[4]xf32>) -> ()
  "test.some_use"(%tile_3) : (vector<[4]x[4]xf32>) -> ()
  "test.some_use"(%tile_4) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

//  CHECK-LIVE-RANGE-LABEL: @tile_live_ins
//        CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//        CHECK-LIVE-RANGE: ^bb0:
//   CHECK-LIVE-RANGE-NEXT: S  arm_sme.get_tile
//   CHECK-LIVE-RANGE-NEXT: |S arm_sme.zero
//   CHECK-LIVE-RANGE-NEXT: EE cf.br
//   CHECK-LIVE-RANGE-NEXT: ^bb1:
//   CHECK-LIVE-RANGE-NEXT: || test.dummy
//   CHECK-LIVE-RANGE-NEXT: || test.dummy
//   CHECK-LIVE-RANGE-NEXT: EE cf.br
//   CHECK-LIVE-RANGE-NEXT: ^bb2:
//   CHECK-LIVE-RANGE-NEXT: || test.dummy
//   CHECK-LIVE-RANGE-NEXT: || test.dummy
//   CHECK-LIVE-RANGE-NEXT: EE cf.br
//   CHECK-LIVE-RANGE-NEXT: ^bb3:
//   CHECK-LIVE-RANGE-NEXT: E| test.some_use
//   CHECK-LIVE-RANGE-NEXT:  E test.some_use

// CHECK-LABEL: @tile_live_ins
func.func @tile_live_ins()
{
  // CHECK: arm_sme.get_tile {tile_id = 0 : i32} : vector<[4]x[4]xf32>
  // CHECK: arm_sme.zero {tile_id = 1 : i32} : vector<[4]x[4]xf32>
  %tile_1 = arm_sme.get_tile : vector<[4]x[4]xf32>
  %tile_2 = arm_sme.zero : vector<[4]x[4]xf32>
  cf.br ^bb1
^bb1:
  "test.dummy"(): () -> ()
  "test.dummy"(): () -> ()
  cf.br ^bb2
^bb2:
  "test.dummy"(): () -> ()
  "test.dummy"(): () -> ()
  cf.br ^bb3
^bb3:
  "test.some_use"(%tile_1) : (vector<[4]x[4]xf32>) -> ()
  "test.some_use"(%tile_2) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

// This is basically the same test as tile_live_ins but shows that the order of
// the blocks within the source does not relate to the liveness, which is based
// on successors and predecessors (not textual order).
//
// So %tile_1 is live on the path bb0 -> bb2 -> bb1 (and dies in bb1). The
// 'hole' when looking at the live range dump comes from the textual order
// (and would disappear if bb1 was moved before bb2 in the source).
//
// When looking at the live range dump (outside of straight-line code) it
// normally makes more sense to consider blocks in isolation (and how they
// relate to the CFG).

//  CHECK-LIVE-RANGE-LABEL: @non_sequential_live_ins
//        CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//        CHECK-LIVE-RANGE: ^bb0:
//   CHECK-LIVE-RANGE-NEXT: S  arm_sme.get_tile
//   CHECK-LIVE-RANGE-NEXT: |  test.dummy
//   CHECK-LIVE-RANGE-NEXT: E  cf.br
//   CHECK-LIVE-RANGE-NEXT: ^bb1:
//   CHECK-LIVE-RANGE-NEXT: E| test.some_use
//   CHECK-LIVE-RANGE-NEXT:  | test.dummy
//   CHECK-LIVE-RANGE-NEXT:  E cf.br
//   CHECK-LIVE-RANGE-NEXT: ^bb2:
//   CHECK-LIVE-RANGE-NEXT: |S arm_sme.zero
//   CHECK-LIVE-RANGE-NEXT: || test.dummy
//   CHECK-LIVE-RANGE-NEXT: EE cf.cond_br
//   CHECK-LIVE-RANGE-NEXT: ^bb3:
//   CHECK-LIVE-RANGE-NEXT:  | test.dummy
//   CHECK-LIVE-RANGE-NEXT:  E test.some_use
//   CHECK-LIVE-RANGE-NEXT:    func.return

// CHECK-LABEL: @non_sequential_live_ins
func.func @non_sequential_live_ins(%cond: i1) {
  // CHECK: arm_sme.get_tile {tile_id = 0 : i32} : vector<[4]x[4]xf32>
  // CHECK: arm_sme.zero {tile_id = 1 : i32} : vector<[4]x[4]xf32>
  %tile_1 = arm_sme.get_tile : vector<[4]x[4]xf32>
  "test.dummy"(): () -> ()
  cf.br ^bb2
^bb1:
  "test.some_use"(%tile_1) : (vector<[4]x[4]xf32>) -> ()
  "test.dummy"(): () -> ()
  cf.br ^bb3
^bb2:
  %tile_2 = arm_sme.zero : vector<[4]x[4]xf32>
  "test.dummy"(): () -> ()
  cf.cond_br %cond, ^bb1, ^bb3
^bb3:
  "test.dummy"(): () -> ()
  "test.some_use"(%tile_2) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

//  CHECK-LIVE-RANGE-LABEL: @non_overlapping_branches
//        CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//        CHECK-LIVE-RANGE: ^bb1:
//   CHECK-LIVE-RANGE-NEXT: S arm_sme.zero
//   CHECK-LIVE-RANGE-NEXT: | arm_sme.copy_tile
//   CHECK-LIVE-RANGE-NEXT: E cf.br
//   CHECK-LIVE-RANGE-NEXT: ^bb2:
//   CHECK-LIVE-RANGE-NEXT: S arm_sme.get_tile
//   CHECK-LIVE-RANGE-NEXT: | arm_sme.copy_tile
//   CHECK-LIVE-RANGE-NEXT: E cf.br

// CHECK-LABEL: @non_overlapping_branches
func.func @non_overlapping_branches(%cond: i1) {
  // CHECK: arm_sme.zero {tile_id = 0 : i32} : vector<[4]x[4]xf32>
  // CHECK: arm_sme.get_tile {tile_id = 0 : i32} : vector<[4]x[4]xf32>
  %tile = scf.if %cond -> vector<[4]x[4]xf32> {
    // ^bb1:
    %zero = arm_sme.zero : vector<[4]x[4]xf32>
    scf.yield %zero : vector<[4]x[4]xf32>
  } else {
    // ^bb2:
    %undef = arm_sme.get_tile : vector<[4]x[4]xf32>
    scf.yield %undef : vector<[4]x[4]xf32>
  }
  "test.some_use"(%tile) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

// Here %vecA and %vecB are not merged into the same live range (as they are unknown values).
// This means that %vecA and %vecB are both allocated to different tiles (which is not legal).

// expected-note@below {{tile operand is: <block argument> of type 'vector<[4]x[4]xf32>'}}
func.func @overlapping_branches(%cond: i1, %vecA: vector<[4]x[4]xf32>, %vecB: vector<[4]x[4]xf32>) {
  // expected-error@below {{op tile operand allocated to different SME virtial tile (move required)}}
  %tile = scf.if %cond -> vector<[4]x[4]xf32> {
    scf.yield %vecA : vector<[4]x[4]xf32>
  } else {
    scf.yield %vecB : vector<[4]x[4]xf32>
  }
  "test.some_use"(%tile) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

//  CHECK-LIVE-RANGE-LABEL: @run_out_of_tiles_but_avoid_spill
//        CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//        CHECK-LIVE-RANGE: ^bb2:
//   CHECK-LIVE-RANGE-NEXT: |S    arm_sme.copy_tile
//   CHECK-LIVE-RANGE-NEXT: ||S   arm_sme.copy_tile
//   CHECK-LIVE-RANGE-NEXT: |||S  arm_sme.copy_tile
//   CHECK-LIVE-RANGE-NEXT: ||||S arm_sme.copy_tile
//   CHECK-LIVE-RANGE-NEXT: EEEEE cf.br

// Note in the live ranges (above) there is five tile values, but we only have four tiles.
// There is no 'real' spill as we spill the `arm_sme.zero` but are then able to clone it
// at each of its uses.

// CHECK-LABEL: @run_out_of_tiles_but_avoid_spill
func.func @run_out_of_tiles_but_avoid_spill(%a: vector<[4]xf32>, %b: vector<[4]xf32>, %c: vector<[4]xf32>, %d: vector<[4]xf32>) {
  %init = arm_sme.zero : vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  // Live = %init
  scf.for %i = %c0 to %c10 step %c1 {
    // CHECK: arm_sme.zero {tile_id = 1 : i32}
    // CHECK: arm_sme.zero {tile_id = 2 : i32}
    // CHECK: arm_sme.zero {tile_id = 3 : i32}
    // CHECK: arm_sme.zero {tile_id = 0 : i32}
    %tile_a, %tile_b, %tile_c, %tile_d = scf.for %j = %c0 to %c10 step %c1
      iter_args(%iter_a = %init, %iter_b = %init, %iter_c = %init, %iter_d = %init)
        -> (vector<[4]x[4]xf32>, vector<[4]x[4]xf32> , vector<[4]x[4]xf32> , vector<[4]x[4]xf32>) {
        // ^bb2:
        // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 1 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
        // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 2 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
        // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 3 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
        // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 0 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
        %new_a = arm_sme.move_vector_to_tile_slice %a, %iter_a, %i : vector<[4]xf32> into vector<[4]x[4]xf32>
        %new_b = arm_sme.move_vector_to_tile_slice %b, %iter_b, %i : vector<[4]xf32> into vector<[4]x[4]xf32>
        %new_c = arm_sme.move_vector_to_tile_slice %c, %iter_c, %i : vector<[4]xf32> into vector<[4]x[4]xf32>
        %new_d = arm_sme.move_vector_to_tile_slice %d, %iter_d, %i : vector<[4]xf32> into vector<[4]x[4]xf32>
        scf.yield %new_a, %new_b, %new_c, %new_d : vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>
    }
    // Live = %init, %tile_a, %tile_b, %tile_c, %tile_d (out of tiles!)
    // This should be resolved by duplicating the arm_sme.zero (from folding
    // arm_sme.copy_tile operations inserted by the tile allocator).
    "test.some_use"(%tile_a) : (vector<[4]x[4]xf32>) -> ()
    "test.some_use"(%tile_b) : (vector<[4]x[4]xf32>) -> ()
    "test.some_use"(%tile_c) : (vector<[4]x[4]xf32>) -> ()
    "test.some_use"(%tile_d) : (vector<[4]x[4]xf32>) -> ()
  }
  return
}

// -----

// We should be able to avoid spills like this, but logic handling this case is
// not implemented yet. Note tile ID >= 16 means a spill/in-memory tile.

//  CHECK-LIVE-RANGE-LABEL: @avoidable_spill
//        CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//        CHECK-LIVE-RANGE: ^bb2:
//   CHECK-LIVE-RANGE-NEXT: ||     test.some_use
//   CHECK-LIVE-RANGE-NEXT: ||S    arm_sme.move_vector_to_tile_slice
//   CHECK-LIVE-RANGE-NEXT: |||S   arm_sme.move_vector_to_tile_slice
//   CHECK-LIVE-RANGE-NEXT: ||||S  arm_sme.move_vector_to_tile_slice
//   CHECK-LIVE-RANGE-NEXT: |||||S arm_sme.move_vector_to_tile_slice
//   CHECK-LIVE-RANGE-NEXT: ||E||| test.some_use
//   CHECK-LIVE-RANGE-NEXT: || E|| test.some_use
//   CHECK-LIVE-RANGE-NEXT: ||  E| test.some_use
//   CHECK-LIVE-RANGE-NEXT: ||   E test.some_use
//   CHECK-LIVE-RANGE-NEXT: ||     arith.addi
//   CHECK-LIVE-RANGE-NEXT: EE     cf.br

// Note in the live ranges (above) there is two constant live-ins (first two ranges),
// which gives six overlapping live ranges (at the point where %tile_d is defined).
// The allocator currently will spill the first constant (which results in a real
// spill at it's use), however, this could be avoided by using the knowledge that
// at the first "test.some_use" there's actually only two live ranges (so we can
// fix this be duplicating the constant).

// CHECK-LABEL: @avoidable_spill
func.func @avoidable_spill(%a: vector<[4]xf32>, %b: vector<[4]xf32>, %c: vector<[4]xf32>, %d: vector<[4]xf32>) {
  // CHECK: arm_sme.zero {tile_id = 16 : i32} : vector<[4]x[4]xf32>
  %zero = arm_sme.zero : vector<[4]x[4]xf32>
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %i = %c0 to %c10 step %c1 {
    // So spilled here (unnecessarily).
    // The arm_sme.zero op could be moved into the loop to avoid this.
    "test.some_use"(%zero) : (vector<[4]x[4]xf32>) -> ()
    %tile_a = arm_sme.move_vector_to_tile_slice %a, %tile, %c0 : vector<[4]xf32> into vector<[4]x[4]xf32>
    %tile_b = arm_sme.move_vector_to_tile_slice %b, %tile, %c0 : vector<[4]xf32> into vector<[4]x[4]xf32>
    %tile_c = arm_sme.move_vector_to_tile_slice %c, %tile, %c0 : vector<[4]xf32> into vector<[4]x[4]xf32>
    %tile_d = arm_sme.move_vector_to_tile_slice %d, %tile, %c0 : vector<[4]xf32> into vector<[4]x[4]xf32>
    // %zero is still live here (due the the backedge)
    "test.some_use"(%tile_a) : (vector<[4]x[4]xf32>) -> ()
    "test.some_use"(%tile_b) : (vector<[4]x[4]xf32>) -> ()
    "test.some_use"(%tile_c) : (vector<[4]x[4]xf32>) -> ()
    "test.some_use"(%tile_d) : (vector<[4]x[4]xf32>) -> ()
  }
  return
}

// -----

// This test is a follow up to the test of the same name in `tile-allocation-copies.mlir`.
// This shows the live ranges (which are why we need to split the conditional branch).

//  CHECK-LIVE-RANGE-LABEL: @cond_branch_with_backedge
//        CHECK-LIVE-RANGE: ^bb1:
//   CHECK-LIVE-RANGE-NEXT:  ||| |           arith.cmpi
//   CHECK-LIVE-RANGE-NEXT:  EEE E           cf.cond_br
//
//   CHECK-LIVE-RANGE-NEXT: ^[[BB3_COPIES:[[:alnum:]]+]]:
//   CHECK-LIVE-RANGE-NEXT:  ||| ES          arm_sme.copy_tile
//   CHECK-LIVE-RANGE-NEXT:  E||  |S         arm_sme.copy_tile
//   CHECK-LIVE-RANGE-NEXT:   E|  ||S        arm_sme.copy_tile
//   CHECK-LIVE-RANGE-NEXT:    E  |||S       arm_sme.copy_tile
//   CHECK-LIVE-RANGE-NEXT:       EEEE       cf.br
//
// It is important to note that the first three live ranges in ^bb1 do not end
// at the `cf.cond_br` they are live-out via the backedge bb1 -> bb2 -> bb1.
// This means that if we placed the `arm_sme.tile_copies` before the `cf.cond_br`
// then those live ranges would not end at the copies, resulting in unwanted
// overlapping live ranges (and hence tile spills).
//
// With the conditional branch split and the copies placed in the BB3_COPIES
// block the first three live ranges end at the copy operations (as the
// BB3_COPIES block is on the path out of the loop and has no backedge). This
// means there is no overlaps and the live ranges all merge, as shown below.
//
//        CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//        CHECK-LIVE-RANGE: ^bb1:
//   CHECK-LIVE-RANGE-NEXT: |||| arith.cmpi
//   CHECK-LIVE-RANGE-NEXT: EEEE cf.cond_br
//
//   CHECK-LIVE-RANGE-NEXT: ^[[BB3_COPIES]]:
//   CHECK-LIVE-RANGE-NEXT: |||| arm_sme.copy_tile
//   CHECK-LIVE-RANGE-NEXT: |||| arm_sme.copy_tile
//   CHECK-LIVE-RANGE-NEXT: |||| arm_sme.copy_tile
//   CHECK-LIVE-RANGE-NEXT: |||| arm_sme.copy_tile
//   CHECK-LIVE-RANGE-NEXT: EEEE cf.br

// CHECK-LABEL: @cond_branch_with_backedge
// CHECK-NOT: tile_id = 16
// CHECK: arm_sme.get_tile {tile_id = 0 : i32} : vector<[4]x[4]xf32>
// CHECK: arm_sme.get_tile {tile_id = 1 : i32} : vector<[4]x[4]xf32>
// CHECK: arm_sme.get_tile {tile_id = 2 : i32} : vector<[4]x[4]xf32>
// CHECK: arm_sme.get_tile {tile_id = 3 : i32} : vector<[4]x[4]xf32>
// CHECK: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 0 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
// CHECK-NOT: tile_id = 16
func.func @cond_branch_with_backedge(%slice: vector<[4]xf32>) {
  %tileA = arm_sme.get_tile : vector<[4]x[4]xf32>
  %tileB = arm_sme.get_tile : vector<[4]x[4]xf32>
  %tileC = arm_sme.get_tile : vector<[4]x[4]xf32>
  %tileD = arm_sme.get_tile : vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  // Live here: %tileA, %tileB, %tileC, %tileD
  cf.br ^bb1(%c0, %tileA : index, vector<[4]x[4]xf32>)
^bb1(%currentIndex: index, %iterTile: vector<[4]x[4]xf32>):
  %continueLoop = arith.cmpi slt, %currentIndex, %c10 : index
  // Live here: %iterTile, %tileB, %tileC, %tileD
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
