// Invoke the test control-flow sink pass to test the utilities.
// RUN: mlir-opt -test-control-flow-sink %s | FileCheck %s

// CHECK-LABEL: func @test_sink
func @test_sink() {
  %0 = "test.sink_me"() : () -> i32
  // CHECK-NEXT: test.sink_target
  "test.sink_target"() ({
    // CHECK-NEXT: %[[V0:.*]] = "test.sink_me"() {was_sunk = 0 : i32}
    // CHECK-NEXT: "test.use"(%[[V0]])
    "test.use"(%0) : (i32) -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @test_sink_first_region_only
func @test_sink_first_region_only() {
  %0 = "test.sink_me"() {first} : () -> i32
  // CHECK-NEXT: %[[V1:.*]] = "test.sink_me"() {second}
  %1 = "test.sink_me"() {second} : () -> i32
  // CHECK-NEXT: test.sink_target
  "test.sink_target"() ({
    // CHECK-NEXT: %[[V0:.*]] = "test.sink_me"() {first, was_sunk = 0 : i32}
    // CHECK-NEXT: "test.use"(%[[V0]])
    "test.use"(%0) : (i32) -> ()
  }, {
    "test.use"(%1) : (i32) -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @test_sink_targeted_op_only
func @test_sink_targeted_op_only() {
  %0 = "test.sink_me"() : () -> i32
  // CHECK-NEXT: %[[V1:.*]] = "test.dont_sink_me"
  %1 = "test.dont_sink_me"() : () -> i32
  // CHECK-NEXT: test.sink_target
  "test.sink_target"() ({
    // CHECK-NEXT: %[[V0:.*]] = "test.sink_me"
    // CHECK-NEXT: "test.use"(%[[V0]], %[[V1]])
    "test.use"(%0, %1) : (i32, i32) -> ()
  }) : () -> ()
  return
}
