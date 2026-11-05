// RUN: mlir-opt %s -int-range-optimizations | FileCheck %s

// Each operation has a defined zero result with 32-bit index values.
// Overflow at 64 bits must not produce poison or a different low 32 bits.

// CHECK-LABEL: @add_nsw
// CHECK: %[[R:.*]] = arith.constant -9223372036854775808 : index
// CHECK: return %[[R]]
func.func @add_nsw() -> index {
  %a = arith.constant 9223372036854775807 : index
  %b = arith.constant 1 : index
  %r = arith.addi %a, %b overflow<nsw> : index
  return %r : index
}

// CHECK-LABEL: @sub_nuw
// CHECK: %[[R:.*]] = arith.constant -4294967296 : index
// CHECK: return %[[R]]
func.func @sub_nuw() -> index {
  %a = arith.constant 0 : index
  %b = arith.constant 4294967296 : index
  %r = arith.subi %a, %b overflow<nuw> : index
  return %r : index
}

// CHECK-LABEL: @mul_nsw
// CHECK: %[[R:.*]] = arith.constant -9223372036854775808 : index
// CHECK: return %[[R]]
func.func @mul_nsw() -> index {
  %a = arith.constant 4611686018427387904 : index
  %b = arith.constant 2 : index
  %r = arith.muli %a, %b overflow<nsw> : index
  return %r : index
}

// CHECK-LABEL: @mul_nuw
// CHECK: %[[R:.*]] = arith.muli {{.*}} overflow<nuw> : index
// CHECK: return %[[R]]
func.func @mul_nuw() -> index {
  %a = arith.constant 4294967296 : index
  %b = arith.constant 4294967296 : index
  %r = arith.muli %a, %b overflow<nuw> : index
  return %r : index
}

// CHECK-LABEL: @shl_nsw
// CHECK: %[[R:.*]] = arith.constant -9223372036854775808 : index
// CHECK: return %[[R]]
func.func @shl_nsw() -> index {
  %a = arith.constant 4611686018427387904 : index
  %b = arith.constant 1 : index
  %r = arith.shli %a, %b overflow<nsw> : index
  return %r : index
}

// CHECK-LABEL: @vector_add
// CHECK: %[[R:.*]] = arith.constant dense<-9223372036854775808> : vector<2xindex>
// CHECK: return %[[R]]
func.func @vector_add() -> vector<2xindex> {
  %a = arith.constant dense<9223372036854775807> : vector<2xindex>
  %b = arith.constant dense<1> : vector<2xindex>
  %r = arith.addi %a, %b overflow<nsw> : vector<2xindex>
  return %r : vector<2xindex>
}

// CHECK-LABEL: @fixed_width_overflow
// CHECK: %[[R:.*]] = ub.poison : i64
// CHECK: return %[[R]]
func.func @fixed_width_overflow() -> i64 {
  %a = arith.constant 9223372036854775807 : i64
  %b = arith.constant 1 : i64
  %r = arith.addi %a, %b overflow<nsw> : i64
  return %r : i64
}

// CHECK-LABEL: @poison_operand
// CHECK: %[[R:.*]] = ub.poison : index
// CHECK: return %[[R]]
func.func @poison_operand(%x: index) -> index {
  %p = ub.poison : index
  %r = arith.addi %x, %p overflow<nsw> : index
  return %r : index
}

// The 64-bit result is defined even though the 32-bit operation overflows.
// CHECK-LABEL: @defined_at_64
// CHECK: %[[R:.*]] = arith.constant 2147483648 : index
// CHECK: return %[[R]]
func.func @defined_at_64() -> index {
  %a = arith.constant 2147483647 : index
  %b = arith.constant 1 : index
  %r = arith.addi %a, %b overflow<nsw> : index
  return %r : index
}
