// RUN: mlir-opt -test-dead-code-analysis 2>&1 %s | FileCheck %s

// An `scf.break` nested in an `scf.if` is a region-branch predecessor of the
// enclosing `scf.execute_region` (an early exit), alongside the normal
// `scf.yield`. This exercises the early-exit generalization of
// RegionBranchOpInterface: the break targets a non-immediate ancestor.

// CHECK-LABEL: test_early_exit:
// CHECK: op_preds: (all) predecessors:
// CHECK-DAG:   scf.yield %{{.*}} : i32
// CHECK-DAG:   scf.break %{{.*}}, %{{.*}} : i32
func.func @test_early_exit(%cond: i1, %a: i32, %b: i32) -> i32 {
  %0 = scf.execute_region -> i32 {
  ^bb0(%tok: token):
    scf.if %cond {
      scf.break %tok, %a : i32
    }
    scf.yield %b : i32
  } {tag = "test_early_exit"}
  return %0 : i32
}

// An `scf.break` that is the immediate terminator of the `scf.execute_region`
// is a (sole) region-branch predecessor of that op.

// CHECK-LABEL: test_immediate_break:
// CHECK: op_preds: (all) predecessors:
// CHECK-NEXT:   scf.break %{{.*}}, %{{.*}} : i32
func.func @test_immediate_break(%a: i32) -> i32 {
  %0 = scf.execute_region -> i32 {
  ^bb0(%tok: token):
    scf.break %tok, %a : i32
  } {tag = "test_immediate_break"}
  return %0 : i32
}

// When *all* terminators of an `scf.execute_region` break to a further
// enclosing op, the inner op's result is never produced: it has no op
// predecessors at all (only its body is entered). The break is attributed to
// the outer op instead.

// CHECK-LABEL: test_all_break_outer_inner:
// CHECK: region_preds: (all) predecessors:
// CHECK-NOT: op_preds
// CHECK-LABEL: test_all_break_outer:
// CHECK: op_preds: (all) predecessors:
// CHECK-DAG:   scf.yield %{{.*}} : i32
// CHECK-DAG:   scf.break %{{.*}}, %{{.*}} : i32
func.func @test_all_break_outer(%a: i32) -> i32 {
  %0 = scf.execute_region -> i32 {
  ^bb0(%tok_outer: token):
    %1 = scf.execute_region -> i32 {
    ^bb1(%tok_inner: token):
      scf.break %tok_outer, %a : i32
    } {tag = "test_all_break_outer_inner"}
    scf.yield %1 : i32
  } {tag = "test_all_break_outer"}
  return %0 : i32
}

// A break that targets the *outer* `scf.execute_region` passes through (and
// skips) the inner one. It is therefore a predecessor of the outer op, but
// *not* of the inner op (whose only exit is its own yield).

// CHECK-LABEL: test_inner:
// CHECK: op_preds: (all) predecessors:
// CHECK-NEXT:   scf.yield %{{.*}} : i32
// CHECK-NOT:   scf.break
// CHECK-LABEL: test_outer:
// CHECK: op_preds: (all) predecessors:
// CHECK-DAG:   scf.yield %{{.*}} : i32
// CHECK-DAG:   scf.break %{{.*}}, %{{.*}} : i32
func.func @test_outer_early_exit(%cond: i1, %a: i32, %b: i32, %c: i32) -> i32 {
  %0 = scf.execute_region -> i32 {
  ^bb0(%tok_outer: token):
    %1 = scf.execute_region -> i32 {
    ^bb1(%tok_inner: token):
      scf.if %cond {
        scf.break %tok_outer, %a : i32
      }
      scf.yield %b : i32
    } {tag = "test_inner"}
    scf.yield %c : i32
  } {tag = "test_outer"}
  return %0 : i32
}
