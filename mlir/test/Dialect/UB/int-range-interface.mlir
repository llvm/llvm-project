// RUN: mlir-opt --int-range-optimizations %s | FileCheck %s

// CHECK-LABEL: func @poison
// CHECK: test.reflect_bounds {smax = -1 : si32, smin = 0 : si32, umax = 0 : ui32, umin = 1 : ui32}
func.func @poison() -> i32 {
    %0 = ub.poison : i32
    %1 = test.reflect_bounds %0 : i32
    func.return %1 : i32
}

// CHECK-LABEL: func @poison_i1
// CHECK: test.reflect_bounds {smax = -1 : si1, smin = 0 : si1, umax = 0 : ui1, umin = 1 : ui1}
func.func @poison_i1() -> i1 {
    %0 = ub.poison : i1
    %1 = test.reflect_bounds %0 : i1
    func.return %1 : i1
}

// CHECK-LABEL: func @poison_non_int
// Check it doesn't crash.
func.func @poison_non_int() -> f32 {
    %0 = ub.poison : f32
    func.return %0 : f32
}
