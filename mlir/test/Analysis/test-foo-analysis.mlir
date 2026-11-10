// RUN: mlir-opt -split-input-file -pass-pipeline='builtin.module(func.func(test-foo-analysis))' %s 2>&1 | FileCheck %s

// CHECK-LABEL: function: @test_default_init
func.func @test_default_init() -> () {
  // CHECK: a -> 0
  "test.foo"() {tag = "a"} : () -> ()
  return
}

// -----

// CHECK-LABEL: function: @test_one_join
func.func @test_one_join() -> () {
  // CHECK: a -> 0
  "test.foo"() {tag = "a"} : () -> ()
  // CHECK: b -> 1
  "test.foo"() {tag = "b", foo = 1 : ui64} : () -> ()
  return
}

// -----

// CHECK-LABEL: function: @test_two_join
func.func @test_two_join() -> () {
  // CHECK: a -> 0
  "test.foo"() {tag = "a"} : () -> ()
  // CHECK: b -> 1
  "test.foo"() {tag = "b", foo = 1 : ui64} : () -> ()
  // CHECK: c -> 3
  "test.foo"() {tag = "c", foo = 2 : ui64} : () -> ()
  return
}

// -----

// CHECK-LABEL: function: @test_fork
func.func @test_fork() -> () {
  // CHECK: init -> 1
  "test.branch"() [^bb0, ^bb1] {tag = "init", foo = 1 : ui64} : () -> ()

^bb0:
  // CHECK: a -> 3
  "test.branch"() [^bb2] {tag = "a", foo = 2 : ui64} : () -> ()

^bb1:
  // CHECK: b -> 5
  "test.branch"() [^bb2] {tag = "b", foo = 4 : ui64} : () -> ()

^bb2:
  // CHECK: end -> 7
  "test.foo"() {tag = "end"} : () -> ()
  return

}

// -----

// CHECK-LABEL: function: @test_simple_loop
func.func @test_simple_loop() -> () {
  // CHECK: init -> 1
  "test.branch"() [^bb0] {tag = "init", foo = 1 : ui64} : () -> ()

^bb0:
  // CHECK: a -> 3
  "test.foo"() {tag = "a", foo = 3 : ui64} : () -> ()
  "test.branch"() [^bb0, ^bb1] : () -> ()

^bb1:
  // CHECK: end -> 3
  "test.foo"() {tag = "end"} : () -> ()
  return
}

// -----

// CHECK-LABEL: function: @test_double_loop
func.func @test_double_loop() -> () {
  // CHECK: init -> 2
  "test.branch"() [^bb0] {tag = "init", foo = 2 : ui64} : () -> ()

^bb0:
  // CHECK: a -> 7
  "test.foo"() {tag = "a", foo = 3 : ui64} : () -> ()
  "test.branch"() [^bb0, ^bb1] : () -> ()

^bb1:
  // CHECK: b -> 7
  "test.foo"() {tag = "b", foo = 5 : ui64} : () -> ()
  "test.branch"() [^bb0, ^bb2] : () -> ()

^bb2:
  // CHECK: end -> 7
  "test.foo"() {tag = "end"} : () -> ()
  return
}

// -----

// The join block is enqueued once per predecessor while `initialize` walks the
// rest of the blocks. The solver collapses those into a single work item, so
// the join block is visited exactly once (in addition to the initialization
// visit). Without worklist deduplication it would be visited once per
// predecessor.

// CHECK-LABEL: function: @wide_fan_in
func.func @wide_fan_in() {
  "test.branch"() [^bb1, ^bb2, ^bb3, ^bb4, ^bb5] : () -> ()

^bb0:
  // CHECK: join block visits -> 2
  // CHECK: join -> 31
  "test.foo"() {tag = "join"} : () -> ()
  return

^bb1:
  "test.branch"() [^bb0] {foo = 1 : ui64} : () -> ()

^bb2:
  "test.branch"() [^bb0] {foo = 2 : ui64} : () -> ()

^bb3:
  "test.branch"() [^bb0] {foo = 4 : ui64} : () -> ()

^bb4:
  "test.branch"() [^bb0] {foo = 8 : ui64} : () -> ()

^bb5:
  "test.branch"() [^bb0] {foo = 16 : ui64} : () -> ()
}
