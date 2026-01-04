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

// -----

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
// CHECK: %[[cast2:.*]] = builtin.unrealized_conversion_cast %[[cast0]] : i32 to i1
// CHECK: %[[extsi:.*]] = arith.extsi %[[cast2]] : i1 to i64
// CHECK: %[[result:.*]] = arith.addi %[[arg0]], %[[extsi]] : i64
// CHECK: return %[[result]] : i64

func.func @liveBifurcation(%arg0: i64) -> i64 {
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to i32
    %1 = builtin.unrealized_conversion_cast %0 : i32 to i64
    %2 = builtin.unrealized_conversion_cast %0 : i32 to i1
    %3 = arith.extsi %2 : i1 to i64
    %4 = arith.addi %1, %3 : i64
    return %4 : i64
}

// -----

// CHECK-LABEL: func @deadNToOneCast(
// CHECK-NEXT:    return
func.func @deadNToOneCast(%arg0: index, %arg1: index) {
    %0 = builtin.unrealized_conversion_cast %arg0, %arg1 : index, index to i64
    return
}

// -----

// CHECK-LABEL: func @swappingOperands(
// CHECK-SAME:      %[[arg0:.*]]: index, %[[arg1:.*]]: index
// CHECK:         %[[cast1:.*]]:2 = builtin.unrealized_conversion_cast %[[arg0]], %[[arg1]]
// CHECK:         %[[cast2:.*]]:2 = builtin.unrealized_conversion_cast %[[cast1]]#1, %[[cast1]]#0
// CHECK:         %[[cast3:.*]]:2 = builtin.unrealized_conversion_cast %[[cast2]]#0, %[[cast2]]#1
// CHECK:         return %[[cast3]]#0, %[[cast3]]#1
func.func @swappingOperands(%arg0: index, %arg1: index) -> (index, index) {
    %0:2 = builtin.unrealized_conversion_cast %arg0, %arg1 : index, index to i64, i64
    %1:2 = builtin.unrealized_conversion_cast %0#1, %0#0 : i64, i64 to i32, i32
    %2:2 = builtin.unrealized_conversion_cast %1#0, %1#1 : i32, i32 to index, index
    return %2#0, %2#1 : index, index
}

// -----

// CHECK-LABEL: func @matchingOperands(
// CHECK-SAME:      %[[arg0:.*]]: index, %[[arg1:.*]]: index
// CHECK:         return %[[arg0]], %[[arg1]]
func.func @matchingOperands(%arg0: index, %arg1: index) -> (index, index) {
    %0:2 = builtin.unrealized_conversion_cast %arg0, %arg1 : index, index to i64, i64
    %1:3 = builtin.unrealized_conversion_cast %0#0, %0#1 : i64, i64 to i32, i32, i32
    %2:2 = builtin.unrealized_conversion_cast %1#0, %1#1, %1#2 : i32, i32, i32 to index, index
    return %2#0, %2#1 : index, index
}

// -----

// CHECK-LABEL: func @emptyCast()
// CHECK:         %[[cast:.*]] = builtin.unrealized_conversion_cast to index
// CHECK:         return %[[cast]]
func.func @emptyCast() -> index {
    %0 = builtin.unrealized_conversion_cast to index
    return %0 : index
}

// -----

// CHECK-LABEL: test.graph_region
//  CHECK-NEXT:   "test.return"() : () -> ()
test.graph_region {
  %0 = builtin.unrealized_conversion_cast %2 : i32 to i64
  %1 = builtin.unrealized_conversion_cast %0 : i64 to i16
  %2 = builtin.unrealized_conversion_cast %1 : i16 to i32
  "test.return"() : () -> ()
}

// -----

// CHECK-LABEL: test.graph_region
//  CHECK-NEXT:   %[[cast0:.*]] = builtin.unrealized_conversion_cast %[[cast2:.*]] : i32 to i64
//  CHECK-NEXT:   %[[cast1:.*]] = builtin.unrealized_conversion_cast %[[cast0]] : i64 to i16
//  CHECK-NEXT:   %[[cast2]] = builtin.unrealized_conversion_cast %[[cast1]] : i16 to i32
//  CHECK-NEXT:   "test.user"(%[[cast2]]) : (i32) -> ()
//  CHECK-NEXT:   "test.return"() : () -> ()
test.graph_region {
  %0 = builtin.unrealized_conversion_cast %2 : i32 to i64
  %1 = builtin.unrealized_conversion_cast %0 : i64 to i16
  %2 = builtin.unrealized_conversion_cast %1 : i16 to i32
  "test.user"(%2) : (i32) -> ()
  "test.return"() : () -> ()
}

// -----

// CHECK-LABEL: test.graph_region
//  CHECK-NEXT:   "test.return"() : () -> ()
test.graph_region {
  %0 = builtin.unrealized_conversion_cast %0 : i32 to i32
  "test.return"() : () -> ()
}

// -----

// CHECK-LABEL: test.graph_region
//  CHECK-NEXT:   %[[c0:.*]] = arith.constant
//  CHECK-NEXT:   %[[cast:.*]]:2 = builtin.unrealized_conversion_cast %[[c0]], %[[cast]]#1 : i32, i32 to i32, i32
//  CHECK-NEXT:   "test.user"(%[[cast]]#0) : (i32) -> ()
//  CHECK-NEXT:   "test.return"() : () -> ()
test.graph_region {
  %cst = arith.constant 0 : i32
  %0, %1 = builtin.unrealized_conversion_cast %cst, %1 : i32, i32 to i32, i32
  "test.user"(%0) : (i32) -> ()
  "test.return"() : () -> ()
}
