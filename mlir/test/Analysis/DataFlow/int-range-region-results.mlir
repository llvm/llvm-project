// RUN: mlir-opt %s -int-range-optimizations | FileCheck %s

// Both inferred values follow a forwarded value. Their indices in the block
// and result list differ from their indices in the non-successor input list.
// The input range grows from [3, 7] to [3, 30] after the region is visited.
// Both the inferred block argument and the inferred result must update.
// CHECK-LABEL: func.func @widened_region_argument_and_result
func.func @widened_region_argument_and_result(%cond: i1) {
  %forwarded = arith.constant 42 : i32
  %delayed = scf.if %cond -> i32 {
    %early = test.with_bounds <umin = 3 : i32, umax = 7 : i32,
                                smin = 3 : i32, smax = 7 : i32> : i32
    scf.yield %early : i32
  } else {
    %nested = scf.execute_region -> i32 {
      %late = test.with_bounds <umin = 20 : i32, umax = 30 : i32,
                               smin = 20 : i32, smax = 30 : i32> : i32
      scf.yield %late : i32
    }
    scf.yield %nested : i32
  }
  %result:2 = "test.with_bounds_region_branch"(%forwarded, %delayed) ({
  ^bb0(%arg0: i32, %arg1: i32):
    // CHECK: test.reflect_bounds <umin = 42 : ui32, umax = 42 : ui32, smin = 42 : si32, smax = 42 : si32>
    %a = test.reflect_bounds %arg0 : i32
    // CHECK: test.reflect_bounds <umin = 3 : ui32, umax = 30 : ui32, smin = 3 : si32, smax = 30 : si32>
    %b = test.reflect_bounds %arg1 : i32
    "test.isolated_region_yield"(%arg0) : (i32) -> ()
  }) : (i32, i32) -> (i32, i32)
  // CHECK: test.reflect_bounds <umin = 42 : ui32, umax = 42 : ui32, smin = 42 : si32, smax = 42 : si32>
  %a = test.reflect_bounds %result#0 : i32
  // CHECK: test.reflect_bounds <umin = 3 : ui32, umax = 30 : ui32, smin = 3 : si32, smax = 30 : si32>
  %b = test.reflect_bounds %result#1 : i32
  return
}
