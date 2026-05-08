// RUN: mlir-opt -pass-pipeline='builtin.module(func.func(test-staged-analyses))' %s | FileCheck %s

// CHECK-LABEL: func.func @linear()
func.func @linear() {
  // CHECK: "test.foo"() {bar_state = true, foo = 1 : ui64, foo_state = 1 : i64, tag = "annotate"} : () -> ()
  "test.foo"() {tag = "annotate", foo = 1 : ui64} : () -> ()
  // CHECK: "test.foo"() {bar_state = true, foo = 2 : ui64, foo_state = 3 : i64, tag = "annotate"} : () -> ()
  "test.foo"() {tag = "annotate", foo = 2 : ui64} : () -> ()
  // CHECK: "test.foo"() {bar_state = true, foo_state = 3 : i64, tag = "annotate"} : () -> ()
  "test.foo"() {tag = "annotate"} : () -> ()
  return
}

// This demonstrates why `BarAnalysis` should be run only after `FooAnalysis`
// converges.
//
// Under the current `FooAnalysis` implementation:
//   - entry op after-state is 0 xor 7 = 7
//   - bb0 terminator after-state is 7 xor 1 = 6
//   - when the join block is first visited, only bb0 has contributed, so the
//     join op transiently sees 6 xor 2 = 4
//   - once the other predecessor arrives, revisiting the join updates the
//     final staged `foo_state` to 7 for the first op in the join block and it
//     stays 7 for the following op
//
// But if a non-staged `BarAnalysis` observed bb2 after only bb0 had reached
// it, bb2's first tagged op would transiently see 6 xor 2 = 4 and latch
// `bar_state = false`, poisoning later points. The staged run below must use
// only the converged `FooState`, so `bar_state` stays true.
//
// CHECK-LABEL: func.func @requires_staged_bar()
func.func @requires_staged_bar() {
  // CHECK: "test.branch"()[^bb{{[0-9]+}}, ^bb{{[0-9]+}}] {bar_state = true, foo = 7 : ui64, foo_state = 7 : i64, tag = "annotate"} : () -> ()
  "test.branch"() [^bb0, ^bb2] {tag = "annotate", foo = 7 : ui64} : () -> ()

^bb0:
  // CHECK: "test.branch"()[^bb{{[0-9]+}}] {bar_state = true, foo = 1 : ui64, foo_state = 6 : i64, tag = "annotate"} : () -> ()
  "test.branch"() [^bb1] {tag = "annotate", foo = 1 : ui64} : () -> ()

^bb1:
  // CHECK: "test.foo"() {bar_state = true, foo = 2 : ui64, foo_state = 7 : i64, tag = "annotate"} : () -> ()
  "test.foo"() {tag = "annotate", foo = 2 : ui64} : () -> ()
  // CHECK: "test.foo"() {bar_state = true, foo_state = 7 : i64, tag = "annotate"} : () -> ()
  "test.foo"() {tag = "annotate"} : () -> ()
  return

^bb2:
  // CHECK: "test.branch"()[^bb{{[0-9]+}}] {bar_state = true, foo = 2 : ui64, foo_state = 5 : i64, tag = "annotate"} : () -> ()
  "test.branch"() [^bb1] {tag = "annotate", foo = 2 : ui64} : () -> ()
}
