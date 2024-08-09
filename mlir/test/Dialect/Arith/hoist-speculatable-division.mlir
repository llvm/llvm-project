// RUN: mlir-opt -loop-invariant-code-motion -split-input-file %s | FileCheck %s

// Verify a division by a non-zero constant is hoisted out of the loop.

// CHECK-LABEL: func @match_non_zero_constant
func.func @match_non_zero_constant(%arg0: i32) {
  %lb = arith.constant 0 : index
  %ub = arith.constant 10 : index
  %step = arith.constant 1 : index
  // CHECK: %[[CST0:.*]] = arith.constant 0 : i32
  %cst0 = arith.constant 0 : i32
  // CHECK: %[[CST1:.*]] = arith.constant 1 : i32
  %cst1 = arith.constant 1 : i32
  // CHECK: = arith.divui %{{.*}}, %[[CST1]]
  // CHECK: scf.for
  scf.for %idx = %lb to %ub step %step {
    // CHECK: = arith.divui %{{.*}}, %[[CST0]]
    %0 = arith.divui %arg0, %cst0 : i32
    %1 = arith.divui %arg0, %cst1 : i32
  }
  return
}

// -----

// Verify a division by a non-zero integer whose range is known due to the
// InferIntRangeInterface is hoisted out of the loop.

// CHECK-LABEL: func @match_integer_range
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
func.func @match_integer_range(%arg0: i8) {
  %lb = arith.constant 0 : index
  %ub = arith.constant 10 : index
  %step = arith.constant 1 : index

  // CHECK: %[[VAL0:.*]] = test.with_bounds
  %0 = test.with_bounds {smax = 127 : i8, smin = -128 : i8, umax = 255 : i8, umin = 0 : i8} : i8
  // CHECK: %[[VAL1:.*]] = test.with_bounds
  %1 = test.with_bounds {smax = 127 : i8, smin = -128 : i8, umax = 255 : i8, umin = 1 : i8} : i8
  // CHECK: %[[VAL2:.*]] = test.with_bounds
  %2 = test.with_bounds {smax = 127 : i8, smin = -128 : i8, umax = 255 : i8, umin = 42 : i8} : i8
  // CHECK: = arith.ceildivui %[[ARG0]], %[[VAL1]]
  // CHECK: = arith.divui %[[ARG0]], %[[VAL2]]
  // CHECK: scf.for
  scf.for %idx = %lb to %ub step %step {
    // CHECK: = arith.divui %[[ARG0]], %[[VAL0]]
    %3 = arith.divui %arg0, %0 : i8
    %4 = arith.ceildivui %arg0, %1 : i8
    %5 = arith.divui %arg0, %2 : i8
    // CHECK: = arith.divui %[[ARG0]], %[[ARG0]]
    %6 = arith.divui %arg0, %arg0 : i8
  }
  return
}
