// RUN: mlir-opt %s -int-range-optimizations | FileCheck %s

// The divisor becomes zero through range inference.

// CHECK-LABEL: @arith_divsi(
// CHECK: %[[P:.*]] = ub.poison : i32
// CHECK: return %[[P]]
func.func @arith_divsi(%x: i32, %n: i32) -> i32 {
  %one = arith.constant 1 : i32
  %bits = arith.andi %x, %one : i32
  %d = arith.remui %bits, %one : i32
  %r = arith.divsi %n, %d : i32
  return %r : i32
}

// CHECK-LABEL: @arith_divui(
// CHECK: %[[P:.*]] = ub.poison : i32
// CHECK: return %[[P]]
func.func @arith_divui(%x: i32, %n: i32) -> i32 {
  %one = arith.constant 1 : i32
  %bits = arith.andi %x, %one : i32
  %d = arith.remui %bits, %one : i32
  %r = arith.divui %n, %d : i32
  return %r : i32
}

// CHECK-LABEL: @arith_ceildivsi(
// CHECK: %[[P:.*]] = ub.poison : i32
// CHECK: return %[[P]]
func.func @arith_ceildivsi(%x: i32, %n: i32) -> i32 {
  %one = arith.constant 1 : i32
  %bits = arith.andi %x, %one : i32
  %d = arith.remui %bits, %one : i32
  %r = arith.ceildivsi %n, %d : i32
  return %r : i32
}

// CHECK-LABEL: @arith_ceildivui(
// CHECK: %[[P:.*]] = ub.poison : i32
// CHECK: return %[[P]]
func.func @arith_ceildivui(%x: i32, %n: i32) -> i32 {
  %one = arith.constant 1 : i32
  %bits = arith.andi %x, %one : i32
  %d = arith.remui %bits, %one : i32
  %r = arith.ceildivui %n, %d : i32
  return %r : i32
}

// CHECK-LABEL: @arith_floordivsi(
// CHECK: %[[P:.*]] = ub.poison : i32
// CHECK: return %[[P]]
func.func @arith_floordivsi(%x: i32, %n: i32) -> i32 {
  %one = arith.constant 1 : i32
  %bits = arith.andi %x, %one : i32
  %d = arith.remui %bits, %one : i32
  %r = arith.floordivsi %n, %d : i32
  return %r : i32
}

// CHECK-LABEL: @arith_remsi(
// CHECK: %[[P:.*]] = ub.poison : i32
// CHECK: return %[[P]]
func.func @arith_remsi(%x: i32, %n: i32) -> i32 {
  %one = arith.constant 1 : i32
  %bits = arith.andi %x, %one : i32
  %d = arith.remui %bits, %one : i32
  %r = arith.remsi %n, %d : i32
  return %r : i32
}

// CHECK-LABEL: @arith_remui(
// CHECK: %[[P:.*]] = ub.poison : i32
// CHECK: return %[[P]]
func.func @arith_remui(%x: i32, %n: i32) -> i32 {
  %one = arith.constant 1 : i32
  %bits = arith.andi %x, %one : i32
  %d = arith.remui %bits, %one : i32
  %r = arith.remui %n, %d : i32
  return %r : i32
}

// CHECK-LABEL: @index_divs(
// CHECK: %[[P:.*]] = ub.poison : index
// CHECK: return %[[P]]
func.func @index_divs(%x: index, %n: index) -> index {
  %one = arith.constant 1 : index
  %bits = arith.andi %x, %one : index
  %d = arith.remui %bits, %one : index
  %r = index.divs %n, %d
  return %r : index
}

// CHECK-LABEL: @index_divu(
// CHECK: %[[P:.*]] = ub.poison : index
// CHECK: return %[[P]]
func.func @index_divu(%x: index, %n: index) -> index {
  %one = arith.constant 1 : index
  %bits = arith.andi %x, %one : index
  %d = arith.remui %bits, %one : index
  %r = index.divu %n, %d
  return %r : index
}

// CHECK-LABEL: @index_ceildivs(
// CHECK: %[[P:.*]] = ub.poison : index
// CHECK: return %[[P]]
func.func @index_ceildivs(%x: index, %n: index) -> index {
  %one = arith.constant 1 : index
  %bits = arith.andi %x, %one : index
  %d = arith.remui %bits, %one : index
  %r = index.ceildivs %n, %d
  return %r : index
}

// CHECK-LABEL: @index_ceildivu(
// CHECK: %[[P:.*]] = ub.poison : index
// CHECK: return %[[P]]
func.func @index_ceildivu(%x: index, %n: index) -> index {
  %one = arith.constant 1 : index
  %bits = arith.andi %x, %one : index
  %d = arith.remui %bits, %one : index
  %r = index.ceildivu %n, %d
  return %r : index
}

// CHECK-LABEL: @index_floordivs(
// CHECK: %[[P:.*]] = ub.poison : index
// CHECK: return %[[P]]
func.func @index_floordivs(%x: index, %n: index) -> index {
  %one = arith.constant 1 : index
  %bits = arith.andi %x, %one : index
  %d = arith.remui %bits, %one : index
  %r = index.floordivs %n, %d
  return %r : index
}

// CHECK-LABEL: @index_rems(
// CHECK: %[[P:.*]] = ub.poison : index
// CHECK: return %[[P]]
func.func @index_rems(%x: index, %n: index) -> index {
  %one = arith.constant 1 : index
  %bits = arith.andi %x, %one : index
  %d = arith.remui %bits, %one : index
  %r = index.rems %n, %d
  return %r : index
}

// CHECK-LABEL: @index_remu(
// CHECK: %[[P:.*]] = ub.poison : index
// CHECK: return %[[P]]
func.func @index_remu(%x: index, %n: index) -> index {
  %one = arith.constant 1 : index
  %bits = arith.andi %x, %one : index
  %d = arith.remui %bits, %one : index
  %r = index.remu %n, %d
  return %r : index
}

// CHECK-LABEL: @ceil_i1_zero(
// CHECK: %[[P:.*]] = ub.poison : i1
// CHECK: return %[[P]]
func.func @ceil_i1_zero(%x: i1) -> i1 {
  %zero = arith.constant 0 : i1
  %r = arith.ceildivsi %x, %zero : i1
  return %r : i1
}

// CHECK-LABEL: @vector_zero_divisor(
// CHECK: %[[P:.*]] = ub.poison : vector<2xi32>
// CHECK: return %[[P]]
func.func @vector_zero_divisor(%x: vector<2xi32>, %n: vector<2xi32>) -> vector<2xi32> {
  %one = arith.constant dense<1> : vector<2xi32>
  %d = arith.remui %x, %one : vector<2xi32>
  %r = arith.divui %n, %d : vector<2xi32>
  return %r : vector<2xi32>
}

// Zero at only one index width must preserve the defined result at the other.
// CHECK-LABEL: @zero_only_at_32(
// CHECK: %[[ZERO:.*]] = index.constant 0
// CHECK: return %[[ZERO]]
func.func @zero_only_at_32() -> index {
  %n = index.constant 7
  %d = index.constant 4294967296
  %r = index.divu %n, %d
  return %r : index
}
