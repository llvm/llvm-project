// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(cse))' | FileCheck %s

// This test is checking that CSE does not remove 'arm_sme.zero/get_tile' ops as
// duplicates.

// CHECK-LABEL: @zero_tile
// CHECK: %[[TILE_0:.*]] = arm_sme.zero : vector<[4]x[4]xi32>
// CHECK: %[[TILE_1:.*]] = arm_sme.zero : vector<[4]x[4]xi32>
// CHECK: "prevent.dce"(%[[TILE_0]]) : (vector<[4]x[4]xi32>) -> ()
// CHECK: "prevent.dce"(%[[TILE_1]]) : (vector<[4]x[4]xi32>) -> ()
func.func @zero_tile() {
  %tile_1 = arm_sme.zero : vector<[4]x[4]xi32>
  %tile_2 = arm_sme.zero : vector<[4]x[4]xi32>
  "prevent.dce"(%tile_1) : (vector<[4]x[4]xi32>) -> ()
  "prevent.dce"(%tile_2) : (vector<[4]x[4]xi32>) -> ()
  return
}

// CHECK-LABEL: @get_tile
// CHECK: %[[TILE_0:.*]] = arm_sme.get_tile : vector<[4]x[4]xi32>
// CHECK: %[[TILE_1:.*]] = arm_sme.get_tile : vector<[4]x[4]xi32>
// CHECK: "prevent.dce"(%[[TILE_0]]) : (vector<[4]x[4]xi32>) -> ()
// CHECK: "prevent.dce"(%[[TILE_1]]) : (vector<[4]x[4]xi32>) -> ()
func.func @get_tile() {
  %tile_1 = arm_sme.get_tile : vector<[4]x[4]xi32>
  %tile_2 = arm_sme.get_tile : vector<[4]x[4]xi32>
  "prevent.dce"(%tile_1) : (vector<[4]x[4]xi32>) -> ()
  "prevent.dce"(%tile_2) : (vector<[4]x[4]xi32>) -> ()
  return
}
