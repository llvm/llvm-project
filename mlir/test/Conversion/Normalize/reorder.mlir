// RUN: mlir-opt %s --normalize --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s

// CHECK-LABEL:   func.func @reorder(
// CHECK-SAME:                   %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %vl14084$51356$ = arith.constant 2 : i32
// CHECK:           %vl15831$funcArg0-vl14084$ = arith.addi %[[ARG0]], %vl14084$51356$ : i32
// CHECK:           %vl14084$187c2$ = arith.constant 6 : i32
// CHECK:           %op27844$vl14084-vl15831$ = arith.addi %vl14084$187c2$, %vl15831$funcArg0-vl14084$ : i32
// CHECK:           %vl14084$4c6ac$ = arith.constant 8 : i32
// CHECK:           %op27844$op27844-vl14084$ = arith.addi %op27844$vl14084-vl15831$, %vl14084$4c6ac$ : i32
// CHECK:           return %op27844$op27844-vl14084$ : i32
// CHECK:         }
func.func @reorder(%a0: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c6 = arith.constant 6 : i32
  %c8 = arith.constant 8 : i32
  %a = arith.addi %a0, %c2 : i32
  %b = arith.addi %a, %c6 : i32
  %c = arith.addi %b, %c8 : i32
  func.return %c : i32
}
