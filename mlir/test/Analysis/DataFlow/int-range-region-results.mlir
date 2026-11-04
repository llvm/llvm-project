// RUN: mlir-opt %s -int-range-optimizations | FileCheck %s

// Both inferred values follow a forwarded value. Their indices in the block
// and result list differ from their indices in the non-successor input list.
// CHECK-LABEL: func.func @delayed_region_argument_and_result
func.func @delayed_region_argument_and_result() {
  %forwarded = arith.constant 42 : i32
  %delayed = scf.execute_region -> i32 {
    %input = test.with_bounds <umin = 3 : i32, umax = 7 : i32,
                              smin = 3 : i32, smax = 7 : i32> : i32
    scf.yield %input : i32
  }
  %result:2 = "test.with_bounds_region_branch"(%forwarded, %delayed) ({
  ^bb0(%arg0: i32, %arg1: i32):
    // CHECK: test.reflect_bounds <umin = 42 : ui32, umax = 42 : ui32, smin = 42 : si32, smax = 42 : si32>
    %a = test.reflect_bounds %arg0 : i32
    // CHECK: test.reflect_bounds <umin = 3 : ui32, umax = 7 : ui32, smin = 3 : si32, smax = 7 : si32>
    %b = test.reflect_bounds %arg1 : i32
    "test.isolated_region_yield"(%arg0) : (i32) -> ()
  }) : (i32, i32) -> (i32, i32)
  // CHECK: test.reflect_bounds <umin = 42 : ui32, umax = 42 : ui32, smin = 42 : si32, smax = 42 : si32>
  %a = test.reflect_bounds %result#0 : i32
  // CHECK: test.reflect_bounds <umin = 3 : ui32, umax = 7 : ui32, smin = 3 : si32, smax = 7 : si32>
  %b = test.reflect_bounds %result#1 : i32
  return
}
