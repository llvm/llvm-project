// RUN: mlir-opt %s -test-commutativity-utils | FileCheck %s

// CHECK-LABEL: @test_small_pattern_1
func.func @test_small_pattern_1(%arg0 : i32) -> i32 {
  // CHECK-NEXT: %[[ARITH_CONST:.*]] = arith.constant
  %0 = arith.constant 45 : i32

  // CHECK-NEXT: %[[TEST_ADD:.*]] = "test.addi"
  %1 = "test.addi"(%arg0, %arg0): (i32, i32) -> i32

  // CHECK-NEXT: %[[ARITH_ADD:.*]] = arith.addi
  %2 = arith.addi %arg0, %arg0 : i32

  // CHECK-NEXT: %[[ARITH_MUL:.*]] = arith.muli
  %3 = arith.muli %arg0, %arg0 : i32

  // CHECK-NEXT: %[[RESULT:.*]] = "test.op_commutative"(%[[ARITH_ADD]], %[[ARITH_MUL]], %[[TEST_ADD]], %[[ARITH_CONST]])
  %result = "test.op_commutative"(%0, %1, %2, %3): (i32, i32, i32, i32) -> i32

  // CHECK-NEXT: return %[[RESULT]]
  return %result : i32
}

// CHECK-LABEL: @test_small_pattern_2
// CHECK-SAME: (%[[ARG0:.*]]: i32
func.func @test_small_pattern_2(%arg0 : i32) -> i32 {
  // CHECK-NEXT: %[[TEST_CONST:.*]] = "test.constant"
  %0 = "test.constant"() {value = 0 : i32} : () -> i32

  // CHECK-NEXT: %[[ARITH_CONST:.*]] = arith.constant
  %1 = arith.constant 0 : i32

  // CHECK-NEXT: %[[ARITH_ADD:.*]] = arith.addi
  %2 = arith.addi %arg0, %arg0 : i32

  // CHECK-NEXT: %[[RESULT:.*]] = "test.op_commutative"(%[[ARG0]], %[[ARITH_ADD]], %[[ARITH_CONST]], %[[TEST_CONST]])
  %result = "test.op_commutative"(%0, %1, %2, %arg0): (i32, i32, i32, i32) -> i32

  // CHECK-NEXT: return %[[RESULT]]
  return %result : i32
}

// CHECK-LABEL: @test_large_pattern
func.func @test_large_pattern(%arg0 : i32, %arg1 : i32) -> i32 {
  // CHECK-NEXT: arith.divsi
  %0 = arith.divsi %arg0, %arg1 : i32

  // CHECK-NEXT: arith.divsi
  %1 = arith.divsi %0, %arg0 : i32

  // CHECK-NEXT: arith.divsi
  %2 = arith.divsi %1, %arg1 : i32

  // CHECK-NEXT: arith.addi
  %3 = arith.addi %1, %arg1 : i32

  // CHECK-NEXT: arith.subi
  %4 = arith.subi %2, %3 : i32

  // CHECK-NEXT: "test.addi"
  %5 = "test.addi"(%arg0, %arg0): (i32, i32) -> i32

  // CHECK-NEXT: %[[VAL6:.*]] = arith.divsi
  %6 = arith.divsi %4, %5 : i32

  // CHECK-NEXT: arith.divsi
  %7 = arith.divsi %1, %arg1 : i32

  // CHECK-NEXT: %[[VAL8:.*]] = arith.muli
  %8 = arith.muli %1, %arg1 : i32

  // CHECK-NEXT: %[[VAL9:.*]] = arith.subi
  %9 = arith.subi %7, %8 : i32

  // CHECK-NEXT: "test.addi"
  %10 = "test.addi"(%arg0, %arg0): (i32, i32) -> i32

  // CHECK-NEXT: %[[VAL11:.*]] = arith.divsi
  %11 = arith.divsi %9, %10 : i32

  // CHECK-NEXT: %[[VAL12:.*]] = arith.divsi
  %12 = arith.divsi %6, %arg1 : i32

  // CHECK-NEXT: arith.subi
  %13 = arith.subi %arg1, %arg0 : i32

  // CHECK-NEXT: "test.op_commutative"(%[[VAL12]], %[[VAL12]], %[[VAL8]], %[[VAL9]])
  %14 = "test.op_commutative"(%12, %9, %12, %8): (i32, i32, i32, i32) -> i32

  // CHECK-NEXT: %[[VAL15:.*]] = arith.divsi
  %15 = arith.divsi %13, %14 : i32

  // CHECK-NEXT: %[[VAL16:.*]] = arith.addi
  %16 = arith.addi %2, %15 : i32

  // CHECK-NEXT: arith.subi
  %17 = arith.subi %16, %arg1 : i32

  // CHECK-NEXT: "test.addi"
  %18 = "test.addi"(%arg0, %arg0): (i32, i32) -> i32

  // CHECK-NEXT: %[[VAL19:.*]] = arith.divsi
  %19 = arith.divsi %17, %18 : i32

  // CHECK-NEXT: "test.addi"
  %20 = "test.addi"(%arg0, %16): (i32, i32) -> i32

  // CHECK-NEXT: %[[VAL21:.*]] = arith.divsi
  %21 = arith.divsi %17, %20 : i32

  // CHECK-NEXT: %[[RESULT:.*]] = "test.op_large_commutative"(%[[VAL16]], %[[VAL19]], %[[VAL19]], %[[VAL21]], %[[VAL6]], %[[VAL11]], %[[VAL15]])
  %result = "test.op_large_commutative"(%16, %6, %11, %15, %19, %21, %19): (i32, i32, i32, i32, i32, i32, i32) -> i32

  // CHECK-NEXT: return %[[RESULT]]
  return %result : i32
}
