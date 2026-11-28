// RUN: mlir-opt %S/cse-overlapping-reads.mlir -cse --mlir-disable-threading -debug-only=cse -o /dev/null 2>&1 | FileCheck %S/cse-overlapping-reads.mlir --check-prefix=EFFECTS --implicit-check-not="CSE checking effects:"
// RUN: mlir-opt %s -cse -allow-unregistered-dialect --mlir-disable-threading -debug-only=cse -o /dev/null 2>&1 | FileCheck %s --implicit-check-not="CSE checking effects:"
// REQUIRES: asserts

// Cache boundary tests, separate from the issue-derived CLIP regression.
// Adjacent duplicate reads must not discard progress for an earlier read.
// Each of the two intervening loads and the addition is classified once.
// CHECK: CSE checking effects: memref.load
// CHECK-NEXT: CSE checking effects: arith.addi
// CHECK-NEXT: CSE checking effects: memref.load
func.func @interleaved_queries(%m: memref<3xi32>) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %a = memref.load %m[%c0] : memref<3xi32>
  %b = memref.load %m[%c1] : memref<3xi32>
  %bd = memref.load %m[%c1] : memref<3xi32>
  %ad = memref.load %m[%c0] : memref<3xi32>
  %sum = arith.addi %b, %bd : i32
  %c = memref.load %m[%c2] : memref<3xi32>
  %cd = memref.load %m[%c2] : memref<3xi32>
  %ad2 = memref.load %m[%c0] : memref<3xi32>
  return %a, %b, %bd, %ad, %sum, %c, %cd, %ad2 : i32, i32, i32, i32, i32, i32, i32, i32
}

// Stop at an unknown effect without inspecting the following addition.
// CHECK-NEXT: CSE checking effects: test.barrier
func.func @early_exit(%m: memref<i32>, %arg: i32) -> i32 {
  %a = memref.load %m[] : memref<i32>
  "test.barrier"() : () -> ()
  %sum = arith.addi %a, %arg : i32
  %b = memref.load %m[] : memref<i32>
  %result = arith.addi %sum, %b : i32
  return %result : i32
}

// Retain each read's progress past disjoint writes.
// CHECK-NEXT: CSE checking effects: test.side_effect_op
// CHECK-NEXT: CSE checking effects: test.side_effect_op
// CHECK-NEXT: CSE checking effects: test.side_effect_op
func.func @disjoint_write_progress() -> (i32, i32, i32, i32) {
  %a = "test.side_effect_op"() {effects = [{effect="read", test_nonaddressable_resource_a}]} : () -> i32
  "test.side_effect_op"() {effects = [{effect="write", test_nonaddressable_resource_b}]} : () -> i32
  %a1 = "test.side_effect_op"() {effects = [{effect="read", test_nonaddressable_resource_a}]} : () -> i32
  "test.side_effect_op"() {effects = [{effect="write", test_nonaddressable_resource_b}]} : () -> i32
  %a2 = "test.side_effect_op"() {effects = [{effect="read", test_nonaddressable_resource_a}]} : () -> i32
  "test.side_effect_op"() {effects = [{effect="write", test_nonaddressable_resource_b}]} : () -> i32
  %a3 = "test.side_effect_op"() {effects = [{effect="read", test_nonaddressable_resource_a}]} : () -> i32
  return %a, %a1, %a2, %a3 : i32, i32, i32, i32
}
