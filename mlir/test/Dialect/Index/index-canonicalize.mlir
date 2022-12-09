// RUN: mlir-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @add
func.func @add() -> (index, index) {
  %0 = index.constant 1
  %1 = index.constant 2100
  %2 = index.constant 3000000001
  %3 = index.constant 4000002100
  // Folds normally.
  %4 = index.add %0, %1
  // Folds even though values exceed INT32_MAX.
  %5 = index.add %2, %3

  // CHECK-DAG: %[[A:.*]] = index.constant 2101
  // CHECK-DAG: %[[B:.*]] = index.constant 7000002101
  // CHECK: return %[[A]], %[[B]]
  return %4, %5 : index, index
}

// CHECK-LABEL: @add_overflow
func.func @add_overflow() -> (index, index) {
  %0 = index.constant 2000000000
  %1 = index.constant 8000000000000000000
  // Folds normally.
  %2 = index.add %0, %0
  // Folds and overflows.
  %3 = index.add %1, %1

  // CHECK-DAG: %[[A:.*]] = index.constant 4{{0+}}
  // CHECK-DAG: %[[B:.*]] = index.constant -2446{{[0-9]+}}
  // CHECK: return %[[A]], %[[B]]
  return %2, %3 : index, index
}

// CHECK-LABEL: @sub
func.func @sub() -> index {
  %0 = index.constant -2000000000
  %1 = index.constant 3000000000
  %2 = index.sub %0, %1
  // CHECK: %[[A:.*]] = index.constant -5{{0+}}
  // CHECK: return %[[A]]
  return %2 : index
}

// CHECK-LABEL: @mul
func.func @mul() -> index {
  %0 = index.constant 8000000002000000000
  %1 = index.constant 2
  %2 = index.mul %0, %1
  // CHECK: %[[A:.*]] = index.constant -2446{{[0-9]+}}
  // CHECK: return %[[A]]
  return %2 : index
}

// CHECK-LABEL: @divs
func.func @divs() -> index {
  %0 = index.constant -2
  %1 = index.constant 0x200000000
  %2 = index.divs %1, %0
  // CHECK: %[[A:.*]] = index.constant -429{{[0-9]+}}
  // CHECK: return %[[A]]
  return %2 : index
}

// CHECK-LABEL: @divs_nofold
func.func @divs_nofold() -> (index, index) {
  %0 = index.constant 0
  %1 = index.constant 0x100000000
  %2 = index.constant 2

  // Divide by zero.
  // CHECK: index.divs
  %3 = index.divs %2, %0
  // 32-bit result differs from 64-bit.
  // CHECK: index.divs
  %4 = index.divs %1, %2

  return %3, %4 : index, index
}

// CHECK-LABEL: @divu
func.func @divu() -> index {
  %0 = index.constant -2
  %1 = index.constant 0x200000000
  %2 = index.divu %1, %0
  // CHECK: %[[A:.*]] = index.constant 0
  // CHECK: return %[[A]]
  return %2 : index
}

// CHECK-LABEL: @divu_nofold
func.func @divu_nofold() -> (index, index) {
  %0 = index.constant 0
  %1 = index.constant 0x100000000
  %2 = index.constant 2

  // Divide by zero.
  // CHECK: index.divu
  %3 = index.divu %2, %0
  // 32-bit result differs from 64-bit.
  // CHECK: index.divu
  %4 = index.divu %1, %2

  return %3, %4 : index, index
}

// CHECK-LABEL: @ceildivs
func.func @ceildivs() -> (index, index, index) {
  %c0 = index.constant 0
  %c2 = index.constant 2
  %c5 = index.constant 5

  // CHECK-DAG: %[[A:.*]] = index.constant 0
  %0 = index.ceildivs %c0, %c5

  // CHECK-DAG: %[[B:.*]] = index.constant 1
  %1 = index.ceildivs %c2, %c5

  // CHECK-DAG: %[[C:.*]] = index.constant 3
  %2 = index.ceildivs %c5, %c2

  // CHECK: return %[[A]], %[[B]], %[[C]]
  return %0, %1, %2 : index, index, index
}

// CHECK-LABEL: @ceildivs_neg
func.func @ceildivs_neg() -> index {
  %c5 = index.constant -5
  %c2 = index.constant 2
  // CHECK: %[[A:.*]] = index.constant -2
  %0 = index.ceildivs %c5, %c2
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @ceildivs_edge
func.func @ceildivs_edge() -> (index, index) {
  %cn1 = index.constant -1
  %cIntMin = index.constant -2147483648
  %cIntMax = index.constant 2147483647

  // The result is 0 on 32-bit.
  // CHECK-DAG: %[[A:.*]] = index.constant 2147483648
  %0 = index.ceildivs %cIntMin, %cn1

  // CHECK-DAG: %[[B:.*]] = index.constant -2147483647
  %1 = index.ceildivs %cIntMax, %cn1

  // CHECK: return %[[A]], %[[B]]
  return %0, %1 : index, index
}

// CHECK-LABEL: @ceildivu
func.func @ceildivu() -> index {
  %0 = index.constant 0x200000001
  %1 = index.constant 2
  // CHECK: %[[A:.*]] = index.constant 429{{[0-9]+}}7
  %2 = index.ceildivu %0, %1
  // CHECK: return %[[A]]
  return %2 : index
}

// CHECK-LABEL: @floordivs
func.func @floordivs() -> index {
  %0 = index.constant -5
  %1 = index.constant 2
  // CHECK: %[[A:.*]] = index.constant -3
  %2 = index.floordivs %0, %1
  // CHECK: return %[[A]]
  return %2 : index
}

// CHECK-LABEL: @floordivs_edge
func.func @floordivs_edge() -> (index, index) {
  %cIntMin = index.constant -2147483648
  %cIntMax = index.constant 2147483647
  %n1 = index.constant -1
  %p1 = index.constant 1

  // CHECK-DAG: %[[A:.*]] = index.constant -2147483648
  // CHECK-DAG: %[[B:.*]] = index.constant -2147483647
  %0 = index.floordivs %cIntMin, %p1
  %1 = index.floordivs %cIntMax, %n1

  // CHECK: return %[[A]], %[[B]]
  return %0, %1 : index, index
}

// CHECK-LABEL: @floordivs_nofold
func.func @floordivs_nofold() -> index {
  %lhs = index.constant 0x100000000
  %c2 = index.constant 2

  // 32-bit result differs from 64-bit.
  // CHECK: index.floordivs
  %0 = index.floordivs %lhs, %c2

  return %0 : index
}

// CHECK-LABEL: @rems
func.func @rems() -> index {
  %lhs = index.constant -5
  %rhs = index.constant 2
  // CHECK: %[[A:.*]] = index.constant -1
  %0 = index.rems %lhs, %rhs
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @rems_nofold
func.func @rems_nofold() -> index {
  %lhs = index.constant 2
  %rhs = index.constant 0x100000001
  // 32-bit result differs from 64-bit.
  // CHECK: index.rems
  %0 = index.rems %lhs, %rhs
  return %0 : index
}

// CHECK-LABEL: @remu
func.func @remu() -> index {
  %lhs = index.constant 2
  %rhs = index.constant -1
  // CHECK: %[[A:.*]] = index.constant 2
  %0 = index.remu %lhs, %rhs
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @remu_nofold
func.func @remu_nofold() -> index {
  %lhs = index.constant 2
  %rhs = index.constant 0x100000001
  // 32-bit result differs from 64-bit.
  // CHECK: index.remu
  %0 = index.remu %lhs, %rhs
  return %0 : index
}

// CHECK-LABEL: @maxs
func.func @maxs() -> index {
  %lhs = index.constant -4
  %rhs = index.constant 2
  // CHECK: %[[A:.*]] = index.constant 2
  %0 = index.maxs %lhs, %rhs
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @maxs_nofold
func.func @maxs_nofold() -> index {
  %lhs = index.constant 1
  %rhs = index.constant 0x100000000
  // 32-bit result differs from 64-bit.
  // CHECK: index.maxs
  %0 = index.maxs %lhs, %rhs
  return %0 : index
}

// CHECK-LABEL: @maxs_edge
func.func @maxs_edge() -> index {
  %lhs = index.constant 1
  %rhs = index.constant 0x100000001
  // Truncated 64-bit result is the same as 32-bit.
  // CHECK: %[[A:.*]] = index.constant 429{{[0-9]+}}
  %0 = index.maxs %lhs, %rhs
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @maxu
func.func @maxu() -> index {
  %lhs = index.constant -1
  %rhs = index.constant 1
  // CHECK: %[[A:.*]] = index.constant -1
  %0 = index.maxu %lhs, %rhs
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @shl
func.func @shl() -> index {
  %lhs = index.constant 128
  %rhs = index.constant 2
  // CHECK: %[[A:.*]] = index.constant 512
  %0 = index.shl %lhs, %rhs
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @shl_32
func.func @shl_32() -> index {
  %lhs = index.constant 1
  %rhs = index.constant 32
  // CHECK: index.shl
  %0 = index.shl %lhs, %rhs
  return %0 : index
}

// CHECK-LABEL: @shl_edge
func.func @shl_edge() -> index {
  %lhs = index.constant 4000000000
  %rhs = index.constant 31
  // CHECK: %[[A:.*]] = index.constant 858{{[0-9]+}}
  %0 = index.shl %lhs, %rhs
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @shrs
func.func @shrs() -> index {
  %lhs = index.constant 128
  %rhs = index.constant 2
  // CHECK: %[[A:.*]] = index.constant 32
  %0 = index.shrs %lhs, %rhs
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @shrs_32
func.func @shrs_32() -> index {
  %lhs = index.constant 4000000000000
  %rhs = index.constant 32
  // CHECK: index.shrs
  %0 = index.shrs %lhs, %rhs
  return %0 : index
}

// CHECK-LABEL: @shrs_nofold
func.func @shrs_nofold() -> index {
  %lhs = index.constant 0x100000000
  %rhs = index.constant 1
  // CHECK: index.shrs
  %0 = index.shrs %lhs, %rhs
  return %0 : index
}

// CHECK-LABEL: @shrs_edge
func.func @shrs_edge() -> index {
  %lhs = index.constant 0x10000000000
  %rhs = index.constant 3
  // CHECK: %[[A:.*]] = index.constant 137{{[0-9]+}}
  %0 = index.shrs %lhs, %rhs
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @shru
func.func @shru() -> index {
  %lhs = index.constant 128
  %rhs = index.constant 2
  // CHECK: %[[A:.*]] = index.constant 32
  %0 = index.shru %lhs, %rhs
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @shru_32
func.func @shru_32() -> index {
  %lhs = index.constant 4000000000000
  %rhs = index.constant 32
  // CHECK: index.shru
  %0 = index.shru %lhs, %rhs
  return %0 : index
}

// CHECK-LABEL: @shru_nofold
func.func @shru_nofold() -> index {
  %lhs = index.constant 0x100000000
  %rhs = index.constant 1
  // CHECK: index.shru
  %0 = index.shru %lhs, %rhs
  return %0 : index
}

// CHECK-LABEL: @shru_edge
func.func @shru_edge() -> index {
  %lhs = index.constant 0x10000000000
  %rhs = index.constant 3
  // CHECK: %[[A:.*]] = index.constant 137{{[0-9]+}}
  %0 = index.shru %lhs, %rhs
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @and
func.func @and() -> index {
  %lhs = index.constant 5
  %rhs = index.constant 1
  // CHECK: %[[A:.*]] = index.constant 1
  %0 = index.and %lhs, %rhs
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @or
func.func @or() -> index {
  %lhs = index.constant 5
  %rhs = index.constant 2
  // CHECK: %[[A:.*]] = index.constant 7
  %0 = index.or %lhs, %rhs
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @xor
func.func @xor() -> index {
  %lhs = index.constant 5
  %rhs = index.constant 1
  // CHECK: %[[A:.*]] = index.constant 4
  %0 = index.xor %lhs, %rhs
  // CHECK: return %[[A]]
  return %0 : index
}

// CHECK-LABEL: @cmp
func.func @cmp() -> (i1, i1, i1, i1) {
  %a = index.constant 0
  %b = index.constant -1
  %c = index.constant -2
  %d = index.constant 4

  %0 = index.cmp slt(%a, %b)
  %1 = index.cmp ugt(%b, %a)
  %2 = index.cmp ne(%d, %a)
  %3 = index.cmp sgt(%b, %a)

  // CHECK-DAG: %[[TRUE:.*]] = index.bool.constant true
  // CHECK-DAG: %[[FALSE:.*]] = index.bool.constant false
  // CHECK: return %[[FALSE]], %[[TRUE]], %[[TRUE]], %[[FALSE]]
  return %0, %1, %2, %3 : i1, i1, i1, i1
}

// CHECK-LABEL: @cmp_nofold
func.func @cmp_nofold() -> i1 {
  %lhs = index.constant 1
  %rhs = index.constant 0x100000000
  // 32-bit result differs from 64-bit.
  // CHECK: index.cmp slt
  %0 = index.cmp slt(%lhs, %rhs)
  return %0 : i1
}

// CHECK-LABEL: @cmp_edge
func.func @cmp_edge() -> i1 {
  %lhs = index.constant 1
  %rhs = index.constant 0x100000002
  // 64-bit result is the same as 32-bit.
  // CHECK: %[[TRUE:.*]] = index.bool.constant true
  %0 = index.cmp slt(%lhs, %rhs)
  // CHECK: return %[[TRUE]]
  return %0 : i1
}
