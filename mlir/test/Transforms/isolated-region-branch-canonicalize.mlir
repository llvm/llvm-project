// RUN: mlir-opt %s --canonicalize | FileCheck %s
// RUN: mlir-opt %s --remove-dead-values --canonicalize | FileCheck %s

func.func private @sink(i32)

// CHECK-LABEL: func.func @dead_input
// CHECK: "test.isolated_region_branch"(%{{.*}}) ({
// CHECK: ^bb0(%[[ARG:.*]]: i32):
// CHECK: call @sink(%[[ARG]])
func.func @dead_input(%x: i32, %y: i32) -> i32 {
  %r = "test.isolated_region_branch"(%x, %y) ({
  ^bb0(%arg: i32, %dead: i32):
    call @sink(%arg) : (i32) -> ()
    %sum = arith.addi %arg, %arg : i32
    "test.isolated_region_yield"(%sum) : (i32) -> ()
  }) : (i32, i32) -> i32
  return %r : i32
}

// CHECK-LABEL: func.func @duplicate_inputs
// CHECK: "test.isolated_region_branch"(%{{.*}}) ({
// CHECK: ^bb0(%[[ARG:.*]]: i32):
// CHECK: call @sink(%[[ARG]])
// CHECK: call @sink(%[[ARG]])
// CHECK: arith.addi %[[ARG]], %[[ARG]]
func.func @duplicate_inputs(%x: i32) -> i32 {
  %r = "test.isolated_region_branch"(%x, %x) ({
  ^bb0(%a: i32, %b: i32):
    call @sink(%a) : (i32) -> ()
    call @sink(%b) : (i32) -> ()
    %sum = arith.addi %a, %b : i32
    "test.isolated_region_yield"(%sum) : (i32) -> ()
  }) : (i32, i32) -> i32
  return %r : i32
}

// CHECK-LABEL: func.func @duplicate_results
// CHECK: %[[R:.*]] = "test.isolated_region_branch"
// CHECK: "test.isolated_region_yield"(%{{.*}}) : (i32) -> ()
// CHECK: return %[[R]], %[[R]] : i32, i32
func.func @duplicate_results(%x: i32) -> (i32, i32) {
  %r:2 = "test.isolated_region_branch"(%x) ({
  ^bb0(%arg: i32):
    %sum = arith.addi %arg, %arg : i32
    "test.isolated_region_yield"(%sum, %sum) : (i32, i32) -> ()
  }) : (i32) -> (i32, i32)
  return %r#0, %r#1 : i32, i32
}

// CHECK-LABEL: func.func @forwarded_result
// CHECK-SAME: %[[X:.*]]: i32
// CHECK: "test.isolated_region_branch"(%[[X]]) ({
// CHECK: ^bb0(%[[ARG:.*]]: i32):
// CHECK: call @sink(%[[ARG]])
// CHECK: "test.isolated_region_yield"() : () -> ()
// CHECK: return %[[X]] : i32
func.func @forwarded_result(%x: i32) -> i32 {
  %r = "test.isolated_region_branch"(%x) ({
  ^bb0(%arg: i32):
    call @sink(%arg) : (i32) -> ()
    "test.isolated_region_yield"(%arg) : (i32) -> ()
  }) : (i32) -> i32
  return %r : i32
}

// An op result can be replaced with an input from an ancestor region. Keep the
// region argument in the isolated body so that its uses do not capture %x.
// CHECK-LABEL: func.func @forwarded_ancestor_result
// CHECK-SAME: %{{.*}}: i1, %[[X:.*]]: i32
// CHECK: %[[SELECTED:.*]] = arith.select %{{.*}}, %[[X]], %{{.*}} : i32
// CHECK: scf.if
// CHECK: "test.isolated_region_branch"(%[[X]]) ({
// CHECK: ^bb0(%[[ARG:.*]]: i32):
// CHECK: call @sink(%[[ARG]])
// CHECK: "test.isolated_region_yield"() : () -> ()
// CHECK: }) : (i32) -> ()
// CHECK: return %[[SELECTED]] : i32
func.func @forwarded_ancestor_result(%cond: i1, %x: i32) -> i32 {
  %r = scf.if %cond -> i32 {
    %forwarded = "test.isolated_region_branch"(%x) ({
    ^bb0(%arg: i32):
      func.call @sink(%arg) : (i32) -> ()
      "test.isolated_region_yield"(%arg) : (i32) -> ()
    }) : (i32) -> i32
    scf.yield %forwarded : i32
  } else {
    %zero = arith.constant 0 : i32
    scf.yield %zero : i32
  }
  return %r : i32
}

// Keep both isolation boundaries when an inner isolated op is nested in an if.
// CHECK-LABEL: func.func @nested_isolation
// CHECK-SAME: %[[COND:.*]]: i1, %[[X:.*]]: i32
// CHECK: "test.isolated_region_branch"(%[[COND]], %[[X]]) ({
// CHECK: ^bb0(%[[C:.*]]: i1, %[[A:.*]]: i32):
// CHECK: scf.if %[[C]]
// CHECK: "test.isolated_region_branch"(%[[A]]) ({
// CHECK: ^bb0(%[[B:.*]]: i32):
// CHECK: call @sink(%[[B]])
// CHECK: "test.isolated_region_yield"() : () -> ()
// CHECK: }) : (i32) -> ()
// CHECK: "test.isolated_region_yield"() : () -> ()
// CHECK: }) : (i1, i32) -> ()
// CHECK: return %[[X]] : i32
func.func @nested_isolation(%cond: i1, %x: i32) -> i32 {
  %r = "test.isolated_region_branch"(%cond, %x) ({
  ^bb0(%c: i1, %a: i32):
    %selected = scf.if %c -> i32 {
      %inner = "test.isolated_region_branch"(%a) ({
      ^bb0(%b: i32):
        func.call @sink(%b) : (i32) -> ()
        "test.isolated_region_yield"(%b) : (i32) -> ()
      }) : (i32) -> i32
      scf.yield %inner : i32
    } else {
      scf.yield %a : i32
    }
    "test.isolated_region_yield"(%selected) : (i32) -> ()
  }) : (i1, i32) -> i32
  return %r : i32
}

// Keep replacement arguments in their own regions. Remove a shared input only
// when its arguments are dead in all regions.
// CHECK-LABEL: func.func @multiple_regions
// CHECK-SAME: %[[X:.*]]: i32,
// CHECK: %[[R:.*]] = "test.isolated_region_branch"(%[[X]]) ({
// CHECK: ^bb0(%[[A:.*]]: i32):
// CHECK: call @sink(%[[A]])
// CHECK: call @sink(%[[A]])
// CHECK: %[[SUM:.*]] = arith.addi %[[A]], %[[A]]
// CHECK: "test.isolated_region_yield"(%[[SUM]]) : (i32) -> ()
// CHECK: ^bb0(%[[B:.*]]: i32):
// CHECK: call @sink(%[[B]])
// CHECK: call @sink(%[[B]])
// CHECK: %[[PRODUCT:.*]] = arith.muli %[[B]], %[[B]]
// CHECK: "test.isolated_region_yield"(%[[PRODUCT]]) : (i32) -> ()
// CHECK: return %[[R]], %[[R]] : i32, i32
func.func @multiple_regions(%x: i32, %y: i32) -> (i32, i32) {
  %r:2 = "test.isolated_region_branch"(%x, %x, %y) ({
  ^bb0(%a: i32, %b: i32, %dead: i32):
    call @sink(%a) : (i32) -> ()
    call @sink(%b) : (i32) -> ()
    %sum = arith.addi %a, %b : i32
    "test.isolated_region_yield"(%sum, %sum) : (i32, i32) -> ()
  }, {
  ^bb0(%a: i32, %b: i32, %dead: i32):
    call @sink(%a) : (i32) -> ()
    call @sink(%b) : (i32) -> ()
    %product = arith.muli %a, %b : i32
    "test.isolated_region_yield"(%product, %product) : (i32, i32) -> ()
  }) : (i32, i32, i32) -> (i32, i32)
  return %r#0, %r#1 : i32, i32
}
