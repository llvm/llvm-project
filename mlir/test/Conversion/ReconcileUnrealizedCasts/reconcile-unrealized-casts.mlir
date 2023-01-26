// RUN: mlir-opt %s -split-input-file -reconcile-unrealized-casts | FileCheck %s

// CHECK-LABEL: @unusedCast
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i64
// CHECK: return %[[arg0]] : i64

func.func @unusedCast(%arg0: i64) -> i64 {
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to i32
    return %arg0 : i64
}

// -----

// CHECK-LABEL: @sameTypes
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i64
// CHECK: return %[[arg0]] : i64

func.func @sameTypes(%arg0: i64) -> i64 {
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to i64
    return %0 : i64
}

// -----

// CHECK-LABEL: @pair
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i64
// CHECK: return %[[arg0]] : i64

func.func @pair(%arg0: i64) -> i64 {
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to i32
    %1 = builtin.unrealized_conversion_cast %0 : i32 to i64
    return %1 : i64
}

// -----

// CHECK-LABEL: @symmetricChain
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i64
// CHECK: return %[[arg0]] : i64

func.func @symmetricChain(%arg0: i64) -> i64 {
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to i32
    %1 = builtin.unrealized_conversion_cast %0 : i32 to i1
    %2 = builtin.unrealized_conversion_cast %1 : i1 to i32
    %3 = builtin.unrealized_conversion_cast %2 : i32 to i64
    return %3 : i64
}

// -----

// CHECK-LABEL: @asymmetricChain
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i64
// CHECK: return %[[arg0]] : i64

func.func @asymmetricChain(%arg0: i64) -> i64 {
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to i32
    %1 = builtin.unrealized_conversion_cast %0 : i32 to i1
    %2 = builtin.unrealized_conversion_cast %1 : i1 to i64
    return %2 : i64
}

// -----

// CHECK-LABEL: @unusedChain
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i64
// CHECK: return %[[arg0]] : i64

func.func @unusedChain(%arg0: i64) -> i64 {
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to i32
    %1 = builtin.unrealized_conversion_cast %0 : i32 to i1
    return %arg0 : i64
}

// -----

// CHECK-LABEL: @bifurcation
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i64
// CHECK: %[[result:.*]] = arith.addi %[[arg0]], %[[arg0]] : i64
// CHECK: return %[[result]] : i64

func.func @bifurcation(%arg0: i64) -> i64 {
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to i32
    %1 = builtin.unrealized_conversion_cast %0 : i32 to i1
    %2 = builtin.unrealized_conversion_cast %1 : i1 to i64
    %3 = builtin.unrealized_conversion_cast %1 : i1 to i32
    %4 = builtin.unrealized_conversion_cast %3 : i32 to i64
    %5 = arith.addi %2, %4 : i64
    return %5 : i64
}

// -----

// CHECK-LABEL: @unusedBifurcation
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i64
// CHECK: %[[result:.*]] = arith.addi %[[arg0]], %[[arg0]] : i64
// CHECK: return %[[result]] : i64

func.func @unusedBifurcation(%arg0: i64) -> i64 {
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to i32
    %1 = builtin.unrealized_conversion_cast %0 : i32 to i1
    %2 = builtin.unrealized_conversion_cast %1 : i1 to i64
    %3 = builtin.unrealized_conversion_cast %0 : i32 to i64
    %4 = arith.addi %arg0, %3 : i64
    return %4 : i64
}
