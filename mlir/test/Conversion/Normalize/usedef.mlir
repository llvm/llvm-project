// RUN: mlir-opt %s --normalize --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s

// CHECK-LABEL:    func.func @use_def(
// CHECK-SAME:         %arg0: i32) -> i32 {
// CHECK:              %vl36495$eafb0$ = arith.constant 4 : i32
// CHECK:              %vl43392$funcArg0-vl36495$ = arith.addi %arg0, %vl36495$eafb0$ : i32
// CHECK:              %vl36495$20b04$ = arith.constant 0 : i32
// CHECK:              %op27844$vl36495-vl43392$ = arith.addi %vl36495$20b04$, %vl43392$funcArg0-vl36495$ : i32
// CHECK:              %op27844$op27844-vl36495$ = arith.addi %op27844$vl36495-vl43392$, %vl36495$eafb0$ : i32
// CHECK:              %op15672$op27844-op27844$ = arith.addi %op27844$op27844-vl36495$, %op27844$vl36495-vl43392$ : i32
// CHECK:              %op15672$op15672-op27844$ = arith.addi %op15672$op27844-op27844$, %op27844$op27844-vl36495$ : i32
// CHECK:              %op15672$op15672-op15672$ = arith.addi %op15672$op15672-op27844$, %op15672$op27844-op27844$ : i32
// CHECK:              %op15672$op15672-op15672$_0 = arith.addi %op15672$op15672-op15672$, %op15672$op15672-op27844$ : i32
// CHECK:              %op15672$op15672-op15672$_1 = arith.addi %op15672$op15672-op15672$, %op15672$op15672-op15672$_0 : i32
// CHECK:              %op15672$op15672-op15672$_2 = arith.addi %op15672$op15672-op15672$, %op15672$op15672-op15672$_1 : i32
// CHECK:              %op15672$op15672-op15672$_3 = arith.addi %op15672$op15672-op15672$_1, %op15672$op15672-op15672$_2 : i32
// CHECK:              return %op15672$op15672-op15672$_3 : i32
// CHECK:            }
module {
  func.func @use_def(%arg0: i32) -> i32 {
    %c0 = arith.constant 4 : i32
    %t  = arith.addi %arg0, %c0 : i32
    %zero = arith.constant 0 : i32
    %t2 = arith.addi %t, %zero : i32
    %t3 = arith.addi %t2, %c0 : i32
    %t4 = arith.addi %t3, %t2 : i32
    %t5 = arith.addi %t4, %t3 : i32
    %t6 = arith.addi %t5, %t4 : i32
    %t7 = arith.addi %t6, %t5 : i32
    %t8 = arith.addi %t7, %t6 : i32
    %t9 = arith.addi %t8, %t6 : i32
    %t10 = arith.addi %t9, %t8 : i32
    return %t10 : i32
  }
}
