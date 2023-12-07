// RUN: mlir-opt -test-int-range-inference -canonicalize %s | FileCheck %s

// Most operations are covered by the `arith` tests, which use the same code
// Here, we add a few tests to ensure the "index can be 32- or 64-bit" handling
// code is operating as expected.

// CHECK-LABEL: func @add_same_for_both
// CHECK: %[[true:.*]] = index.bool.constant true
// CHECK: return %[[true]]
func.func @add_same_for_both(%arg0 : index) -> i1 {
  %c1 = index.constant 1
  %calmostBig = index.constant 0xfffffffe
  %0 = index.minu %arg0, %calmostBig
  %1 = index.add %0, %c1
  %2 = index.cmp uge(%1, %c1)
  func.return %2 : i1
}

// CHECK-LABEL: func @add_unsigned_ov
// CHECK: %[[uge:.*]] = index.cmp uge
// CHECK: return %[[uge]]
func.func @add_unsigned_ov(%arg0 : index) -> i1 {
  %c1 = index.constant 1
  %cu32_max = index.constant 0xffffffff
  %0 = index.minu %arg0, %cu32_max
  %1 = index.add %0, %c1
  // On 32-bit, the add could wrap, so the result doesn't have to be >= 1
  %2 = index.cmp uge(%1, %c1)
  func.return %2 : i1
}

// CHECK-LABEL: func @add_signed_ov
// CHECK: %[[sge:.*]] = index.cmp sge
// CHECK: return %[[sge]]
func.func @add_signed_ov(%arg0 : index) -> i1 {
  %c0 = index.constant 0
  %c1 = index.constant 1
  %ci32_max = index.constant 0x7fffffff
  %0 = index.minu %arg0, %ci32_max
  %1 = index.add %0, %c1
  // On 32-bit, the add could wrap, so the result doesn't have to be positive
  %2 = index.cmp sge(%1, %c0)
  func.return %2 : i1
}

// CHECK-LABEL: func @add_big
// CHECK: %[[true:.*]] = index.bool.constant true
// CHECK: return %[[true]]
func.func @add_big(%arg0 : index) -> i1 {
  %c1 = index.constant 1
  %cmin = index.constant 0x300000000
  %cmax = index.constant 0x30000ffff
  // Note: the order of the clamps matters.
  // If you go max, then min, you infer the ranges [0x300...0, 0xff..ff]
  // and then [0x30...0000, 0x30...ffff]
  // If you switch the order of the below operations, you instead first infer
  // the range [0,0x3...ffff]. Then, the min inference can't constraint
  // this intermediate, since in the 32-bit case we could have, for example
  // trunc(%arg0 = 0x2ffffffff) = 0xffffffff > trunc(0x30000ffff) = 0x0000ffff
  // which means we can't do any inference.
  %0 = index.maxu %arg0, %cmin
  %1 = index.minu %0, %cmax
  %2 = index.add %1, %c1
  %3 = index.cmp uge(%1, %cmin)
  func.return %3 : i1
}
