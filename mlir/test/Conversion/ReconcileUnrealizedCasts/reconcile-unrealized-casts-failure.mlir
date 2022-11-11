// RUN: not mlir-opt %s -split-input-file -mlir-print-ir-after-failure -reconcile-unrealized-casts 2>&1 | FileCheck %s

// CHECK-LABEL: @liveSingleCast
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i32
// CHECK: %[[liveCast:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : i64 to i32
// CHECK: return %[[liveCast]] : i32

func.func @liveSingleCast(%arg0: i64) -> i32 {
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to i32
    return %0 : i32
}

// -----

// CHECK-LABEL: @liveChain
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i32
// CHECK: %[[cast0:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : i64 to i1
// CHECK: %[[cast1:.*]] = builtin.unrealized_conversion_cast %[[cast0]] : i1 to i32
// CHECK: return %[[cast1]] : i32

func.func @liveChain(%arg0: i64) -> i32 {
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to i1
    %1 = builtin.unrealized_conversion_cast %0 : i1 to i32
    return %1 : i32
}

// -----

// CHECK-LABEL: @liveBifurcation
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i64
// CHECK: %[[cast0:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : i64 to i32
// CHECK: %[[cast1:.*]] = builtin.unrealized_conversion_cast %[[cast0]] : i32 to i64
// CHECK: %[[cast2:.*]] = builtin.unrealized_conversion_cast %[[cast0]] : i32 to i1
// CHECK: %[[extsi:.*]] = arith.extsi %[[cast2]] : i1 to i64
// CHECK: %[[result:.*]] = arith.addi %[[cast1]], %[[extsi]] : i64
// CHECK: return %[[result]] : i64

func.func @liveBifurcation(%arg0: i64) -> i64 {
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to i32
    %1 = builtin.unrealized_conversion_cast %0 : i32 to i64
    %2 = builtin.unrealized_conversion_cast %0 : i32 to i1
    %3 = arith.extsi %2 : i1 to i64
    %4 = arith.addi %1, %3 : i64
    return %4 : i64
}
