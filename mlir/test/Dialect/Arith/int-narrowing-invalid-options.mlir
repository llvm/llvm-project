// RUN: not mlir-opt %s --arith-int-narrowing --mlir-print-ir-after-failure 2>&1 \
// RUN:   | FileCheck %s

// RUN: not mlir-opt %s --arith-int-narrowing="int-bitwidths-supported=0" \
// RUN:   --mlir-print-ir-after-failure 2>&1 | FileCheck %s

// Make sure we do not crash on invalid pass options.

// CHECK:       IR Dump After ArithIntNarrowing Failed (arith-int-narrowing)
// CHECK-LABEL: func.func @addi_extsi_i8
func.func @addi_extsi_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extsi %rhs : i8 to i32
  %r = arith.addi %a, %b : i32
  return %r : i32
}
