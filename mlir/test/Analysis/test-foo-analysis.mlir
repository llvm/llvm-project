// RUN: mlir-opt -split-input-file -pass-pipeline='func.func(test-foo-analysis)' %s 2>&1 | FileCheck %s

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
  // CHECK: c -> 0
  "test.foo"() {tag = "c", foo = 1 : ui64} : () -> ()
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
  // CHECK: end -> 6
  "test.foo"() {tag = "end"} : () -> ()
  return

}

// -----

// CHECK-LABEL: function: @test_simple_loop
func.func @test_simple_loop() -> () {
  // CHECK: init -> 1
  "test.branch"() [^bb0] {tag = "init", foo = 1 : ui64} : () -> ()

^bb0:
  // CHECK: a -> 1
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
  // CHECK: a -> 1
  "test.foo"() {tag = "a", foo = 3 : ui64} : () -> ()
  "test.branch"() [^bb0, ^bb1] : () -> ()

^bb1:
  // CHECK: b -> 4
  "test.foo"() {tag = "b", foo = 5 : ui64} : () -> ()
  "test.branch"() [^bb0, ^bb2] : () -> ()

^bb2:
  // CHECK: end -> 4
  "test.foo"() {tag = "end"} : () -> ()
  return
}
