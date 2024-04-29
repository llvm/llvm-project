// RUN: mlir-opt %s -allocate-arm-sme-tiles -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=CHECK-BAD

// This file tests some aspects of liveness issues in the SME tile allocator.
// These tests were designed with a new liveness-based tile allocator in mind,
// with the current tile allocator these tests all give incorrect results (which
// is documented by `CHECK-BAD`).
//
// Currently only the `CHECK-BAD` tests are run (as the new liveness based
// allocator is not yet available -- so all other tests fail).

//       CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//  CHECK-LIVE-RANGE-NEXT: @constant_with_multiple_users
//       CHECK-LIVE-RANGE: ^bb0:
//       CHECK-LIVE-RANGE: S  arm_sme.zero
//  CHECK-LIVE-RANGE-NEXT: |S arm_sme.move_vector_to_tile_slice
//  CHECK-LIVE-RANGE-NEXT: || arm_sme.move_vector_to_tile_slice
//  CHECK-LIVE-RANGE-NEXT: |E test.some_use
//  CHECK-LIVE-RANGE-NEXT: E  test.some_use

// Incorrect result! The second `move_vector_to_tile_slice` overwrites the first (which is still live).
//
// CHECK-BAD-LABEL: @constant_with_multiple_users(
// CHECK-BAD-SAME:                                %[[VECTOR_A:.*]]: vector<[4]xf32>, %[[VECTOR_B:.*]]: vector<[4]xf32>
// CHECK-BAD: %[[ZERO_TILE:.*]] = arm_sme.zero {tile_id = 0 : i32} : vector<[4]x[4]xf32>
// CHECK-BAD: %[[INSERT_TILE_1:.*]] = arm_sme.move_vector_to_tile_slice %[[VECTOR_A]], %[[ZERO_TILE]], %{{.*}} {tile_id = 0 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
// CHECK-BAD: %[[INSERT_TILE_0:.*]] = arm_sme.move_vector_to_tile_slice %[[VECTOR_B]], %[[ZERO_TILE]], %{{.*}} {tile_id = 0 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>

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

//       CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//  CHECK-LIVE-RANGE-NEXT: @value_with_multiple_users
//       CHECK-LIVE-RANGE: ^bb0:
//  CHECK-LIVE-RANGE-NEXT: |S arm_sme.move_vector_to_tile_slice
//  CHECK-LIVE-RANGE-NEXT: || arm_sme.move_vector_to_tile_slice
//  CHECK-LIVE-RANGE-NEXT: |E test.some_use
//  CHECK-LIVE-RANGE-NEXT: E  test.some_use

// (No CHECK-BAD -- the current tile allocator ignores this case)

func.func @value_with_multiple_users(%tile: vector<[4]x[4]xf32>, %a: vector<[4]xf32>, %b: vector<[4]xf32>, %index: index) {
  // A future allocator should error here (as `%`tile would need to be copied).
  %tile_a = arm_sme.move_vector_to_tile_slice %a, %tile, %index : vector<[4]xf32> into vector<[4]x[4]xf32>
  %tile_b = arm_sme.move_vector_to_tile_slice %b, %tile, %index : vector<[4]xf32> into vector<[4]x[4]xf32>
  "test.some_use"(%tile_a) : (vector<[4]x[4]xf32>) -> ()
  "test.some_use"(%tile_b) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

//       CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//  CHECK-LIVE-RANGE-NEXT: @reuse_tiles_after_initial_use
//       CHECK-LIVE-RANGE: ^bb0:
//  CHECK-LIVE-RANGE-NEXT: S        arm_sme.get_tile
//  CHECK-LIVE-RANGE-NEXT: |S       arm_sme.get_tile
//  CHECK-LIVE-RANGE-NEXT: ||S      arm_sme.get_tile
//  CHECK-LIVE-RANGE-NEXT: |||S     arm_sme.get_tile
//  CHECK-LIVE-RANGE-NEXT: ||||     test.dummy
//  CHECK-LIVE-RANGE-NEXT: ||||     test.dummy
//  CHECK-LIVE-RANGE-NEXT: ||||     test.dummy
//  CHECK-LIVE-RANGE-NEXT: E|||     test.some_use
//  CHECK-LIVE-RANGE-NEXT:  E||     test.some_use
//  CHECK-LIVE-RANGE-NEXT:   E|     test.some_use
//  CHECK-LIVE-RANGE-NEXT:    E     test.some_use
//  CHECK-LIVE-RANGE-NEXT:     S    arm_sme.zero
//  CHECK-LIVE-RANGE-NEXT:     |S   arm_sme.zero
//  CHECK-LIVE-RANGE-NEXT:     ||S  arm_sme.zero
//  CHECK-LIVE-RANGE-NEXT:     |||S arm_sme.zero
//  CHECK-LIVE-RANGE-NEXT:     |||| test.dummy
//  CHECK-LIVE-RANGE-NEXT:     |||| test.dummy
//  CHECK-LIVE-RANGE-NEXT:     |||| test.dummy
//  CHECK-LIVE-RANGE-NEXT:     E||| test.some_use
//  CHECK-LIVE-RANGE-NEXT:      E|| test.some_use
//  CHECK-LIVE-RANGE-NEXT:       E| test.some_use
//  CHECK-LIVE-RANGE-NEXT:        E test.some_use

// CHECK-BAD-LABEL: @reuse_tiles_after_initial_use
// CHECK-BAD: arm_sme.get_tile {tile_id = 0 : i32}
// CHECK-BAD: arm_sme.get_tile {tile_id = 1 : i32}
// CHECK-BAD: arm_sme.get_tile {tile_id = 2 : i32}
// CHECK-BAD: arm_sme.get_tile {tile_id = 3 : i32}
//
// -> Spills after the fourth tile (unnecessary):
//
// CHECK-BAD: arm_sme.zero {tile_id = 16 : i32}
// CHECK-BAD: arm_sme.zero {tile_id = 17 : i32}
// CHECK-BAD: arm_sme.zero {tile_id = 18 : i32}
// CHECK-BAD: arm_sme.zero {tile_id = 19 : i32}

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
  // Unnecessary spills:
  // expected-warning @below {{failed to allocate SME virtual tile to operation, all tile operations will go through memory, expect degraded performance}}
  %tile_1 = arm_sme.zero : vector<[4]x[4]xf32>
  // expected-warning @below {{failed to allocate SME virtual tile to operation, all tile operations will go through memory, expect degraded performance}}
  %tile_2 = arm_sme.zero : vector<[4]x[4]xf32>
  // expected-warning @below {{failed to allocate SME virtual tile to operation, all tile operations will go through memory, expect degraded performance}}
  %tile_3 = arm_sme.zero : vector<[4]x[4]xf32>
  // expected-warning @below {{failed to allocate SME virtual tile to operation, all tile operations will go through memory, expect degraded performance}}
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

//       CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//  CHECK-LIVE-RANGE-NEXT: @non_overlapping_branches
//       CHECK-LIVE-RANGE: ^bb1:
//  CHECK-LIVE-RANGE-NEXT: S arm_sme.zero
//  CHECK-LIVE-RANGE-NEXT: | arm_sme.copy_tile
//  CHECK-LIVE-RANGE-NEXT: E cf.br
//  CHECK-LIVE-RANGE-NEXT: ^bb2:
//  CHECK-LIVE-RANGE-NEXT: S arm_sme.get_tile
//  CHECK-LIVE-RANGE-NEXT: | arm_sme.copy_tile
//  CHECK-LIVE-RANGE-NEXT: E cf.br

// Incorrect result! Both branches should yield the result via the same tile.
//
// CHECK-BAD-LABEL: @non_overlapping_branches
// CHECK-BAD: arm_sme.zero {tile_id = 0 : i32} : vector<[4]x[4]xf32>
// CHECK-BAD: arm_sme.get_tile {tile_id = 1 : i32} : vector<[4]x[4]xf32>

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

//       CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
// <deliberately omitted>

// Incorrect result! Everything assigned to tile 0 (which means values that are still live are overwritten).
//
// CHECK-BAD-LABEL: @constant_loop_init_with_multiple_users
// CHECK-BAD: arm_sme.zero {tile_id = 0 : i32} : vector<[4]x[4]xf32>
// CHECK-BAD: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 0 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
// CHECK-BAD: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 0 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>

// CHECK-LABEL: @constant_loop_init_with_multiple_users
func.func @constant_loop_init_with_multiple_users(%a: vector<[4]xf32>, %b: vector<[4]xf32>) {
  // CHECK: arm_sme.zero {tile_id = 0 : i32} : vector<[4]x[4]xf32>
  // CHECK: arm_sme.zero {tile_id = 1 : i32} : vector<[4]x[4]xf32>
  // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 1 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
  // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 0 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init = arm_sme.zero : vector<[4]x[4]xf32>
  %tile_a = scf.for %i = %c0 to %c10 step %c1 iter_args(%iter = %init) -> vector<[4]x[4]xf32> {
    %new_tile = arm_sme.move_vector_to_tile_slice %a, %iter, %i : vector<[4]xf32> into vector<[4]x[4]xf32>
    scf.yield %new_tile : vector<[4]x[4]xf32>
  }
  %tile_b = scf.for %i = %c0 to %c10 step %c1 iter_args(%iter = %init) -> vector<[4]x[4]xf32> {
    %new_tile = arm_sme.move_vector_to_tile_slice %a, %iter, %i : vector<[4]xf32> into vector<[4]x[4]xf32>
    scf.yield %new_tile : vector<[4]x[4]xf32>
  }
  "test.some_use"(%tile_a) : (vector<[4]x[4]xf32>) -> ()
  "test.some_use"(%tile_b) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

//       CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//  CHECK-LIVE-RANGE-NEXT: @run_out_of_tiles_but_avoid_spill
//       CHECK-LIVE-RANGE: ^bb2:
//  CHECK-LIVE-RANGE-NEXT: |S    arm_sme.copy_tile
//  CHECK-LIVE-RANGE-NEXT: ||S   arm_sme.copy_tile
//  CHECK-LIVE-RANGE-NEXT: |||S  arm_sme.copy_tile
//  CHECK-LIVE-RANGE-NEXT: ||||S arm_sme.copy_tile
//  CHECK-LIVE-RANGE-NEXT: EEEEE cf.br

// Note in the live ranges (above) there is five tile values, but we only have four tiles.

// Incorrect result! Everything assigned to tile 0 (which means values that are still live are overwritten).
//
// CHECK-BAD-LABEL: @run_out_of_tiles_but_avoid_spill
// CHECK-BAD: arm_sme.zero {tile_id = 0 : i32}
// CHECK-BAD-COUNT-4: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 0 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>

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

//       CHECK-LIVE-RANGE: ========== Coalesced Live Ranges:
//  CHECK-LIVE-RANGE-NEXT: @avoidable_spill
//       CHECK-LIVE-RANGE: ^bb2:
//  CHECK-LIVE-RANGE-NEXT: ||     test.some_use
//  CHECK-LIVE-RANGE-NEXT: ||S    arm_sme.move_vector_to_tile_slice
//  CHECK-LIVE-RANGE-NEXT: |||S   arm_sme.move_vector_to_tile_slice
//  CHECK-LIVE-RANGE-NEXT: ||||S  arm_sme.move_vector_to_tile_slice
//  CHECK-LIVE-RANGE-NEXT: |||||S arm_sme.move_vector_to_tile_slice
//  CHECK-LIVE-RANGE-NEXT: ||E||| test.some_use
//  CHECK-LIVE-RANGE-NEXT: || E|| test.some_use
//  CHECK-LIVE-RANGE-NEXT: ||  E| test.some_use
//  CHECK-LIVE-RANGE-NEXT: ||   E test.some_use
//  CHECK-LIVE-RANGE-NEXT: ||     arith.addi
//  CHECK-LIVE-RANGE-NEXT: EE     cf.br
//
// Note in the live ranges (above) there is two constant live-ins (first two ranges),
// which gives six overlapping live ranges. The allocator currently will spill the
// first constant (which results in a real spill at it's use), however, this could
// be avoided by using the knowledge that at the first "test.some_use" there's
// actually only two live ranges (so we can fix this be duplicating the constant).

// Incorrect result! Everything other than zero assigned to tile 1 (which means values that are still live are overwritten).
//
// CHECK-BAD-LABEL: @avoidable_spill
// CHECK-BAD: arm_sme.zero {tile_id = 0 : i32}
// CHECK-BAD: arm_sme.get_tile {tile_id = 1 : i32}
// CHECK-BAD-COUNT-4: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 1 : i32}

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
