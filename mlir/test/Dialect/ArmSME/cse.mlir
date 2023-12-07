// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(cse))' | FileCheck %s

// This test is checking that CSE does not remove 'arm_sme.get_tile_id' ops as
// duplicates.
// CHECK-LABEL: @get_tile_id
// CHECK: %[[TILE_ID_0:.*]] = arm_sme.get_tile_id : i32
// CHECK: %[[TILE_ID_1:.*]] = arm_sme.get_tile_id : i32
// CHECK: "prevent.dce"(%[[TILE_ID_0]]) : (i32) -> ()
// CHECK: "prevent.dce"(%[[TILE_ID_1]]) : (i32) -> ()
func.func @get_tile_id() {
  %tile_id_1 = arm_sme.get_tile_id : i32
  %tile_id_2 = arm_sme.get_tile_id : i32
  "prevent.dce"(%tile_id_1) : (i32) -> ()
  "prevent.dce"(%tile_id_2) : (i32) -> ()
  return
}
