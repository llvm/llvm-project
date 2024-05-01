// RUN: mlir-opt %s -allocate-arm-sme-tiles -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=CHECK-BAD

// This file tests some aspects of liveness issues in the SME tile allocator.
// These tests were designed with a new liveness-based tile allocator in mind
// (where the names of test cases make more sense), with the current tile
// allocator these tests all give incorrect results (which is documented by
// `CHECK-BAD`).

// Incorrect result! The second `move_vector_to_tile_slice` overwrites the first (which is still live).
//
// CHECK-BAD-LABEL: @constant_with_multiple_users
// CHECK-BAD: %[[ZERO_TILE:.*]] = arm_sme.zero {tile_id = 0 : i32} : vector<[4]x[4]xf32>
// CHECK-BAD: %[[INSERT_TILE_1:.*]] = arm_sme.move_vector_to_tile_slice %{{.*}} {tile_id = 0 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
// CHECK-BAD: %[[INSERT_TILE_0:.*]] = arm_sme.move_vector_to_tile_slice %{{.*}} {tile_id = 0 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
func.func @constant_with_multiple_users(%a: vector<[4]xf32>, %b: vector<[4]xf32>, %index: index) {
  %zero = arm_sme.zero : vector<[4]x[4]xf32>
  %tile_a = arm_sme.move_vector_to_tile_slice %a, %zero, %index : vector<[4]xf32> into vector<[4]x[4]xf32>
  %tile_b = arm_sme.move_vector_to_tile_slice %b, %zero, %index : vector<[4]xf32> into vector<[4]x[4]xf32>
  "test.some_use"(%tile_a) : (vector<[4]x[4]xf32>) -> ()
  "test.some_use"(%tile_b) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

// (No tile IDs -- the current tile allocator ignores this case)

// CHECK-BAD-LABEL: @value_with_multiple_users
// CHECK-BAD-NOT: tile_id
func.func @value_with_multiple_users(%tile: vector<[4]x[4]xf32>, %a: vector<[4]xf32>, %b: vector<[4]xf32>, %index: index) {
  // A future allocator should error here (as `%tile` would need to be copied).
  %tile_a = arm_sme.move_vector_to_tile_slice %a, %tile, %index : vector<[4]xf32> into vector<[4]x[4]xf32>
  %tile_b = arm_sme.move_vector_to_tile_slice %b, %tile, %index : vector<[4]xf32> into vector<[4]x[4]xf32>
  "test.some_use"(%tile_a) : (vector<[4]x[4]xf32>) -> ()
  "test.some_use"(%tile_b) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

// CHECK-BAD-LABEL: @reuse_tiles_after_initial_use
func.func @reuse_tiles_after_initial_use() {
  // CHECK-BAD: arm_sme.get_tile {tile_id = 0 : i32}
  // CHECK-BAD: arm_sme.get_tile {tile_id = 1 : i32}
  // CHECK-BAD: arm_sme.get_tile {tile_id = 2 : i32}
  // CHECK-BAD: arm_sme.get_tile {tile_id = 3 : i32}
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
  // -> Spills after the fourth tile (unnecessary):
  // CHECK-BAD: arm_sme.zero {tile_id = 16 : i32}
  // CHECK-BAD: arm_sme.zero {tile_id = 17 : i32}
  // CHECK-BAD: arm_sme.zero {tile_id = 18 : i32}
  // CHECK-BAD: arm_sme.zero {tile_id = 19 : i32}
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

// Incorrect result! Both branches should yield the result via the same tile.
//
// CHECK-BAD-LABEL: @non_overlapping_branches
// CHECK-BAD: arm_sme.zero {tile_id = 0 : i32} : vector<[4]x[4]xf32>
// CHECK-BAD: arm_sme.get_tile {tile_id = 1 : i32} : vector<[4]x[4]xf32>
func.func @non_overlapping_branches(%cond: i1) {
  %tile = scf.if %cond -> vector<[4]x[4]xf32> {
    %zero = arm_sme.zero : vector<[4]x[4]xf32>
    scf.yield %zero : vector<[4]x[4]xf32>
  } else {
    %undef = arm_sme.get_tile : vector<[4]x[4]xf32>
    scf.yield %undef : vector<[4]x[4]xf32>
  }
  "test.some_use"(%tile) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

// Incorrect result! Everything assigned to tile 0 (which means values that are still live are overwritten).
//
// CHECK-BAD-LABEL: @constant_loop_init_with_multiple_users
// CHECK-BAD: arm_sme.zero {tile_id = 0 : i32} : vector<[4]x[4]xf32>
// CHECK-BAD: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 0 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
// CHECK-BAD: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 0 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
func.func @constant_loop_init_with_multiple_users(%a: vector<[4]xf32>, %b: vector<[4]xf32>) {
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

// Incorrect result! Everything assigned to tile 0 (which means values that are still live are overwritten).
//
// CHECK-BAD-LABEL: @run_out_of_tiles_but_avoid_spill
// CHECK-BAD: arm_sme.zero {tile_id = 0 : i32}
// CHECK-BAD-COUNT-4: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 0 : i32} : vector<[4]xf32> into vector<[4]x[4]xf32>
func.func @run_out_of_tiles_but_avoid_spill(%a: vector<[4]xf32>, %b: vector<[4]xf32>, %c: vector<[4]xf32>, %d: vector<[4]xf32>) {
  %init = arm_sme.zero : vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %tile_a, %tile_b, %tile_c, %tile_d = scf.for %j = %c0 to %c10 step %c1
      iter_args(%iter_a = %init, %iter_b = %init, %iter_c = %init, %iter_d = %init)
        -> (vector<[4]x[4]xf32>, vector<[4]x[4]xf32> , vector<[4]x[4]xf32> , vector<[4]x[4]xf32>) {
        %new_a = arm_sme.move_vector_to_tile_slice %a, %iter_a, %i : vector<[4]xf32> into vector<[4]x[4]xf32>
        %new_b = arm_sme.move_vector_to_tile_slice %b, %iter_b, %i : vector<[4]xf32> into vector<[4]x[4]xf32>
        %new_c = arm_sme.move_vector_to_tile_slice %c, %iter_c, %i : vector<[4]xf32> into vector<[4]x[4]xf32>
        %new_d = arm_sme.move_vector_to_tile_slice %d, %iter_d, %i : vector<[4]xf32> into vector<[4]x[4]xf32>
        scf.yield %new_a, %new_b, %new_c, %new_d : vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>
    }
    "test.some_use"(%tile_a) : (vector<[4]x[4]xf32>) -> ()
    "test.some_use"(%tile_b) : (vector<[4]x[4]xf32>) -> ()
    "test.some_use"(%tile_c) : (vector<[4]x[4]xf32>) -> ()
    "test.some_use"(%tile_d) : (vector<[4]x[4]xf32>) -> ()
  }
  return
}

// -----

// Incorrect result! Everything other than zero assigned to tile 1 (which means values that are still live are overwritten).
//
// CHECK-BAD-LABEL: @avoidable_spill
// CHECK-BAD: arm_sme.zero {tile_id = 0 : i32}
// CHECK-BAD: arm_sme.get_tile {tile_id = 1 : i32}
// CHECK-BAD-COUNT-4: arm_sme.move_vector_to_tile_slice {{.*}} {tile_id = 1 : i32}
func.func @avoidable_spill(%a: vector<[4]xf32>, %b: vector<[4]xf32>, %c: vector<[4]xf32>, %d: vector<[4]xf32>) {
  %zero = arm_sme.zero : vector<[4]x[4]xf32>
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %i = %c0 to %c10 step %c1 {
    "test.some_use"(%zero) : (vector<[4]x[4]xf32>) -> ()
    %tile_a = arm_sme.move_vector_to_tile_slice %a, %tile, %c0 : vector<[4]xf32> into vector<[4]x[4]xf32>
    %tile_b = arm_sme.move_vector_to_tile_slice %b, %tile, %c0 : vector<[4]xf32> into vector<[4]x[4]xf32>
    %tile_c = arm_sme.move_vector_to_tile_slice %c, %tile, %c0 : vector<[4]xf32> into vector<[4]x[4]xf32>
    %tile_d = arm_sme.move_vector_to_tile_slice %d, %tile, %c0 : vector<[4]xf32> into vector<[4]x[4]xf32>
    "test.some_use"(%tile_a) : (vector<[4]x[4]xf32>) -> ()
    "test.some_use"(%tile_b) : (vector<[4]x[4]xf32>) -> ()
    "test.some_use"(%tile_c) : (vector<[4]x[4]xf32>) -> ()
    "test.some_use"(%tile_d) : (vector<[4]x[4]xf32>) -> ()
  }
  return
}
