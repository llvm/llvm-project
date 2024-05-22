// RUN: mlir-opt %s -topological-sort | FileCheck %s
// RUN: mlir-opt %s -test-topological-sort-analysis -verify-diagnostics | FileCheck %s -check-prefix=CHECK-ANALYSIS

// Test producer is after user.
// CHECK-LABEL: test.graph_region
// CHECK-ANALYSIS-LABEL: test.graph_region
test.graph_region attributes{"root"} {
  // CHECK-NEXT: test.foo
  // CHECK-NEXT: test.baz
  // CHECK-NEXT: test.bar

  // CHECK-ANALYSIS-NEXT: test.foo{{.*}} {pos = 0
  // CHECK-ANALYSIS-NEXT: test.bar{{.*}} {pos = 2
  // CHECK-ANALYSIS-NEXT: test.baz{{.*}} {pos = 1
  %0 = "test.foo"() {selected} : () -> i32
  "test.bar"(%1, %0) {selected} : (i32, i32) -> ()
  %1 = "test.baz"() {selected} : () -> i32
}

// Test cycles.
// CHECK-LABEL: test.graph_region
// CHECK-ANALYSIS-LABEL: test.graph_region
test.graph_region attributes{"root"} {
  // CHECK-NEXT: test.d
  // CHECK-NEXT: test.a
  // CHECK-NEXT: test.c
  // CHECK-NEXT: test.b

  // CHECK-ANALYSIS-NEXT: test.c{{.*}} {pos = 0
  // CHECK-ANALYSIS-NEXT: test.b{{.*}} : (
  // CHECK-ANALYSIS-NEXT: test.a{{.*}} {pos = 2
  // CHECK-ANALYSIS-NEXT: test.d{{.*}} {pos = 1
  %2 = "test.c"(%1) {selected} : (i32) -> i32
  %1 = "test.b"(%0, %2) : (i32, i32) -> i32
  %0 = "test.a"(%3) {selected} : (i32) -> i32
  %3 = "test.d"() {selected} : () -> i32
}

// Test not all scheduled.
// CHECK-LABEL: test.graph_region
// CHECK-ANALYSIS-LABEL: test.graph_region
// expected-error@+1 {{could not schedule all ops}}
test.graph_region attributes{"root"} {
  %0 = "test.a"(%1) {selected} : (i32) -> i32
  %1 = "test.b"(%0) {selected} : (i32) -> i32
}

// CHECK-LABEL: func @test_multiple_blocks
// CHECK-ANALYSIS-LABEL: func @test_multiple_blocks
func.func @test_multiple_blocks() -> (i32) attributes{"root", "ordered"} {
  // CHECK-ANALYSIS-NEXT: test.foo{{.*}} {pos = 0
  %0 = "test.foo"() {selected = 2} : () -> (i32)
  // CHECK-ANALYSIS-NEXT: test.foo
  %1 = "test.foo"() : () -> (i32)
  cf.br ^bb0
^bb0:
  // CHECK-ANALYSIS: test.foo{{.*}} {pos = 1
  %2 = "test.foo"() {selected = 3} : () -> (i32)
  // CHECK-ANALYSIS-NEXT: test.bar{{.*}} {pos = 2
  %3 = "test.bar"(%0, %1, %2) {selected = 0} : (i32, i32, i32) -> (i32)
  cf.br ^bb1 (%2 : i32)
^bb1(%arg0: i32):
  // CHECK-ANALYSIS: test.qux{{.*}} {pos = 3
  %4 = "test.qux"(%arg0, %0) {selected = 1} : (i32, i32) -> (i32)
  return %4 : i32
}

// Test block arguments.
// CHECK-LABEL: test.graph_region
test.graph_region {
// CHECK-NEXT: (%{{.*}}:
^entry(%arg0: i32):
  // CHECK-NEXT: test.foo
  // CHECK-NEXT: test.baz
  // CHECK-NEXT: test.bar
  %0 = "test.foo"(%arg0) : (i32) -> i32
  "test.bar"(%1, %0) : (i32, i32) -> ()
  %1 = "test.baz"(%arg0) : (i32) -> i32
}

// Test implicit block capture (and sort nested region).
// CHECK-LABEL: test.graph_region
func.func @test_graph_cfg() -> () {
  %0 = "test.foo"() : () -> i32
  cf.br ^next(%0 : i32)

^next(%1: i32):
  test.graph_region {
    // CHECK-NEXT: test.foo
    // CHECK-NEXT: test.baz
    // CHECK-NEXT: test.bar
    %2 = "test.foo"(%1) : (i32) -> i32
    "test.bar"(%3, %2) : (i32, i32) -> ()
    %3 = "test.baz"(%0) : (i32) -> i32
  }
  return
}

// Test region ops (and recursive sort).
// CHECK-LABEL: test.graph_region
test.graph_region {
  // CHECK-NEXT: test.baz
  // CHECK-NEXT: test.graph_region attributes {a} {
  // CHECK-NEXT:   test.b
  // CHECK-NEXT:   test.a
  // CHECK-NEXT: }
  // CHECK-NEXT: test.bar
  // CHECK-NEXT: test.foo
  %0 = "test.foo"(%1) : (i32) -> i32
  test.graph_region attributes {a} {
    %a = "test.a"(%b) : (i32) -> i32
    %b = "test.b"(%2) : (i32) -> i32
  }
  %1 = "test.bar"(%2) : (i32) -> i32
  %2 = "test.baz"() : () -> i32
}
