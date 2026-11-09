// RUN: mlir-opt %s --canonicalize | FileCheck %s

func.func private @sink_i32(i32)
func.func private @sink_i64(i64)

// Block arg whose only reachable value is a differently-typed op operand must
// NOT be replaced with the operand.
// CHECK-LABEL: func.func @type_changing_block_arg
// CHECK-SAME: (%[[X:.*]]: i64)
// CHECK: "test.type_changing_region_branch"(%[[X]]) ({
// CHECK: ^bb0(%[[ARG:.*]]: i32):
// CHECK:   call @sink_i32(%[[ARG]])
// CHECK:   "test.type_changing_yield"() : () -> ()
// CHECK: }) : (i64) -> ()
func.func @type_changing_block_arg(%x: i64) {
  "test.type_changing_region_branch"(%x) ({
  ^bb0(%arg: i32):
    func.call @sink_i32(%arg) : (i32) -> ()
    "test.type_changing_yield"() : () -> ()
  }) : (i64) -> ()
  return
}

// Op result whose only reachable value is a differently-typed yielded
// value must NOT be replaced with the yielded value.
// CHECK-LABEL: func.func @type_changing_op_result
// CHECK: %[[C:.*]] = arith.constant 0 : i32
// CHECK: %[[R:.*]] = "test.type_changing_region_branch"() ({
// CHECK:   "test.type_changing_yield"(%[[C]]) : (i32) -> ()
// CHECK: }) : () -> i64
// CHECK: call @sink_i64(%[[R]])
func.func @type_changing_op_result() {
  %c0 = arith.constant 0 : i32
  %r = "test.type_changing_region_branch"() ({
    "test.type_changing_yield"(%c0) : (i32) -> ()
  }) : () -> i64
  func.call @sink_i64(%r) : (i64) -> ()
  return
}

// A same-typed block argument still canonicalizes as before (no regression).
// CHECK-LABEL: func.func @same_type_block_arg
// CHECK-SAME: (%[[X:.*]]: i32)
// CHECK: "test.type_changing_region_branch"() ({
// CHECK:   call @sink_i32(%[[X]])
// CHECK:   "test.type_changing_yield"() : () -> ()
// CHECK: }) : () -> ()
func.func @same_type_block_arg(%x: i32) {
  "test.type_changing_region_branch"(%x) ({
  ^bb0(%arg: i32):
    func.call @sink_i32(%arg) : (i32) -> ()
    "test.type_changing_yield"() : () -> ()
  }) : (i32) -> ()
  return
}

// Duplicate inputs with differing types must NOT be replaced with each other.
// CHECK-LABEL: func.func @duplicate_inputs_different_types
// CHECK-SAME: (%[[X:.*]]: f32)
// CHECK: "test.type_changing_region_branch"(%[[X]], %[[X]]) ({
// CHECK: ^bb0(%[[A:.*]]: i32, %[[B:.*]]: i64):
// CHECK:   call @sink_i32(%[[A]])
// CHECK:   call @sink_i64(%[[B]])
// CHECK:   "test.type_changing_yield"() : () -> ()
// CHECK: }) : (f32, f32) -> ()
func.func @duplicate_inputs_different_types(%x: f32) {
  "test.type_changing_region_branch"(%x, %x) ({
  ^bb0(%a: i32, %b: i64):
    func.call @sink_i32(%a) : (i32) -> ()
    func.call @sink_i64(%b) : (i64) -> ()
    "test.type_changing_yield"() : () -> ()
  }) : (f32, f32) -> ()
  return
}

// Duplicate inputs with the same type alongside a different type can still be deduplicated.
// CHECK-LABEL: func.func @duplicate_inputs_same_type_dedup
// CHECK-SAME: (%[[X:.*]]: f32)
// CHECK: "test.type_changing_region_branch"(%[[X]], %[[X]]) ({
// CHECK: ^bb0(%[[A:.*]]: i64, %[[B:.*]]: i32):
// CHECK:   call @sink_i64(%[[A]])
// CHECK:   call @sink_i32(%[[B]])
// CHECK:   call @sink_i32(%[[B]])
// CHECK:   "test.type_changing_yield"() : () -> ()
// CHECK: }) : (f32, f32) -> ()
func.func @duplicate_inputs_same_type_dedup(%x: f32) {
  "test.type_changing_region_branch"(%x, %x, %x) ({
  ^bb0(%a: i64, %b: i32, %c: i32):
    func.call @sink_i64(%a) : (i64) -> ()
    func.call @sink_i32(%b) : (i32) -> ()
    func.call @sink_i32(%c) : (i32) -> ()
    "test.type_changing_yield"() : () -> ()
  }) : (f32, f32, f32) -> ()
  return
}

// Duplicate op results with differing types must NOT be replaced with each other.
// CHECK-LABEL: func.func @duplicate_results_different_types
// CHECK-SAME: (%[[X:.*]]: i32)
// CHECK: %[[R:.*]]:2 = "test.type_changing_region_branch"() ({
// CHECK:   %[[SUM:.*]] = arith.addi %[[X]], %[[X]] : i32
// CHECK:   "test.type_changing_yield"(%[[SUM]], %[[SUM]]) : (i32, i32) -> ()
// CHECK: }) : () -> (i32, i64)
// CHECK: return %[[R]]#0, %[[R]]#1 : i32, i64
func.func @duplicate_results_different_types(%x: i32) -> (i32, i64) {
  %r:2 = "test.type_changing_region_branch"(%x) ({
  ^bb0(%arg: i32):
    %sum = arith.addi %arg, %arg : i32
    "test.type_changing_yield"(%sum, %sum) : (i32, i32) -> ()
  }) : (i32) -> (i32, i64)
  return %r#0, %r#1 : i32, i64
}

// Two same-typed results plus one differently-typed result receive the same
// yielded value -> the same-typed pair dedupes.
// CHECK-LABEL: func.func @duplicate_results_same_type_dedup
// CHECK-SAME: (%[[X:.*]]: i32)
// CHECK: %[[R:.*]]:2 = "test.type_changing_region_branch"() ({
// CHECK:   %[[SUM:.*]] = arith.addi %[[X]], %[[X]] : i32
// CHECK:   "test.type_changing_yield"(%[[SUM]], %[[SUM]]) : (i32, i32) -> ()
// CHECK: }) : () -> (i64, i32)
// CHECK: return %[[R]]#0, %[[R]]#1, %[[R]]#1 : i64, i32, i32
func.func @duplicate_results_same_type_dedup(%x: i32) -> (i64, i32, i32) {
  %r:3 = "test.type_changing_region_branch"(%x) ({
  ^bb0(%arg: i32):
    %sum = arith.addi %arg, %arg : i32
    "test.type_changing_yield"(%sum, %sum, %sum) : (i32, i32, i32) -> ()
  }) : (i32) -> (i64, i32, i32)
  return %r#0, %r#1, %r#2 : i64, i32, i32
}
