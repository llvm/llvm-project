// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(test-diagnostic-filter{filters=mysource1}))" -split-input-file -o - 2>&1 | FileCheck %s
// This test verifies that diagnostic handler can emit the call stack successfully.

// CHECK-LABEL: Test 'test1'
// CHECK-NEXT: mysource2:1:0: error: test diagnostic
// CHECK-NEXT: mysource3:2:0: note: called from
func.func private @test1() attributes {
  test.loc = loc(callsite("foo"("mysource1":0:0) at callsite("mysource2":1:0 at "mysource3":2:0)))
}

// -----

// CHECK-LABEL: Test 'test2'
// CHECK-NEXT: mysource1:0:0: error: test diagnostic
func.func private @test2() attributes {
  test.loc = loc("mysource1":0:0)
}
