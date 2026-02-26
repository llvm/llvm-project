// RUN: mlir-opt -int-range-optimizations -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @readfirstlane
// CHECK: test.reflect_bounds {smax = 10 : si32, smin = 0 : si32, umax = 10 : ui32, umin = 0 : ui32}
func.func @readfirstlane() -> i32 {
  %0 = test.with_bounds { umin = 0 : ui32, umax = 10 : ui32, smin = 0 : si32, smax = 10 : si32 } : i32
  %1 = rocdl.readfirstlane %0 : i32
  %2 = test.reflect_bounds %1 : i32
  return %2 : i32
}

// -----

// CHECK-LABEL: func @readlane
// CHECK: test.reflect_bounds {smax = 10 : si32, smin = 0 : si32, umax = 10 : ui32, umin = 0 : ui32}
func.func @readlane(%idx: i32) -> i32 {
  %0 = test.with_bounds { umin = 0 : ui32, umax = 10 : ui32, smin = 0 : si32, smax = 10 : si32 } : i32
  %1 = rocdl.readlane %0, %idx : (i32, i32) -> i32
  %2 = test.reflect_bounds %1 : i32
  return %2 : i32
}
