// RUN: mlir-opt -test-generic-ir-visitors-interrupt -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// Walk is interrupted before visiting "foo"
func @main(%arg0: f32) -> f32 {
  %v1 = "foo"() {interrupt_before_all = true} : () -> f32
  %v2 = arith.addf %v1, %arg0 : f32
  return %v2 : f32
}

// CHECK: step 0 op 'builtin.module' before all regions
// CHECK: step 1 op 'func.func' before all regions
// CHECK: step 2 walk was interrupted

// -----

// Walk is interrupted after visiting "foo" (which has a single empty region)
func @main(%arg0: f32) -> f32 {
  %v1 = "foo"() ({ "bar"() : ()-> () }) {interrupt_after_all = true} : () -> f32
  %v2 = arith.addf %v1, %arg0 : f32
  return %v2 : f32
}

// CHECK: step 0 op 'builtin.module' before all regions
// CHECK: step 1 op 'func.func' before all regions
// CHECK: step 2 op 'foo' before all regions
// CHECK: step 3 op 'bar' before all regions
// CHECK: step 4 walk was interrupted

// -----

// Walk is interrupted after visiting "foo"'s 1st region.
func @main(%arg0: f32) -> f32 {
  %v1 = "foo"() ({
    "bar0"() : () -> ()
  }, {
    "bar1"() : () -> ()
  }) {interrupt_after_region = 0} : () -> f32
  %v2 = arith.addf %v1, %arg0 : f32
  return %v2 : f32
}

// CHECK: step 0 op 'builtin.module' before all regions
// CHECK: step 1 op 'func.func' before all regions
// CHECK: step 2 op 'foo' before all regions
// CHECK: step 3 op 'bar0' before all regions
// CHECK: step 4 walk was interrupted


// -----

// Test static filtering.
func @main() {
  "foo"() : () -> ()
  "test.two_region_op"()(
    {"work"() : () -> ()},
    {"work"() : () -> ()}
  ) {interrupt_after_all = true} : () -> ()
  return
}

// CHECK: step 0 op 'builtin.module' before all regions
// CHECK: step 1 op 'func.func' before all regions
// CHECK: step 2 op 'foo' before all regions
// CHECK: step 3 op 'test.two_region_op' before all regions
// CHECK: step 4 op 'work' before all regions
// CHECK: step 5 op 'test.two_region_op' before region #1
// CHECK: step 6 op 'work' before all regions
// CHECK: step 7 walk was interrupted
// CHECK: step 8 op 'test.two_region_op' before all regions
// CHECK: step 9 op 'test.two_region_op' before region #1
// CHECK: step 10 walk was interrupted

// -----

// Test static filtering.
func @main() {
  "foo"() : () -> ()
  "test.two_region_op"()(
    {"work"() : () -> ()},
    {"work"() : () -> ()}
  ) {interrupt_after_region = 0} : () -> ()
  return
}

// CHECK: step 0 op 'builtin.module' before all regions
// CHECK: step 1 op 'func.func' before all regions
// CHECK: step 2 op 'foo' before all regions
// CHECK: step 3 op 'test.two_region_op' before all regions
// CHECK: step 4 op 'work' before all regions
// CHECK: step 5 walk was interrupted
// CHECK: step 6 op 'test.two_region_op' before all regions
// CHECK: step 7 walk was interrupted

// -----
// Test skipping.

// Walk is skipped before visiting "foo".
func @main(%arg0: f32) -> f32 {
  %v1 = "foo"() ({
    "bar0"() : () -> ()
  }, {
    "bar1"() : () -> ()
  }) {skip_before_all = true} : () -> f32
  %v2 = arith.addf %v1, %arg0 : f32
  return %v2 : f32
}

// CHECK: step 0 op 'builtin.module' before all regions
// CHECK: step 1 op 'func.func' before all regions
// CHECK: step 2 op 'arith.addf' before all regions
// CHECK: step 3 op 'func.return' before all regions
// CHECK: step 4 op 'func.func' after all regions
// CHECK: step 5 op 'builtin.module' after all regions

// -----
// Walk is skipped after visiting all regions of "foo".
func @main(%arg0: f32) -> f32 {
  %v1 = "foo"() ({
    "bar0"() : () -> ()
  }, {
    "bar1"() : () -> ()
  }) {skip_after_all = true} : () -> f32
  %v2 = arith.addf %v1, %arg0 : f32
  return %v2 : f32
}

// CHECK: step 0 op 'builtin.module' before all regions
// CHECK: step 1 op 'func.func' before all regions
// CHECK: step 2 op 'foo' before all regions
// CHECK: step 3 op 'bar0' before all regions
// CHECK: step 4 op 'foo' before region #1
// CHECK: step 5 op 'bar1' before all regions
// CHECK: step 6 op 'arith.addf' before all regions
// CHECK: step 7 op 'func.return' before all regions
// CHECK: step 8 op 'func.func' after all regions
// CHECK: step 9 op 'builtin.module' after all regions

// -----
// Walk is skipped after visiting first region of "foo".
func @main(%arg0: f32) -> f32 {
  %v1 = "foo"() ({
    "bar0"() : () -> ()
  }, {
    "bar1"() : () -> ()
  }) {skip_after_region = 0} : () -> f32
  %v2 = arith.addf %v1, %arg0 : f32
  return %v2 : f32
}

// CHECK: step 0 op 'builtin.module' before all regions
// CHECK: step 1 op 'func.func' before all regions
// CHECK: step 2 op 'foo' before all regions
// CHECK: step 3 op 'bar0' before all regions
// CHECK: step 4 op 'arith.addf' before all regions
// CHECK: step 5 op 'func.return' before all regions
// CHECK: step 6 op 'func.func' after all regions
// CHECK: step 7 op 'builtin.module' after all regions
