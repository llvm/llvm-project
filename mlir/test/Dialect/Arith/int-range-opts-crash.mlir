// RUN: mlir-opt -int-range-optimizations %s | FileCheck %s

// CHECK-LABEL: func.func @repro_crash() -> !test.i32 {
func.func @repro_crash() -> !test.i32 {
  %cst = arith.constant 1 : i32
  // CHECK: %[[RES:.*]] = test.region_type_changer
  %0 = "test.region_type_changer"(%cst) ({
  ^bb0(%arg0: i32):
    "test.types_compat_yield"(%arg0) : (i32) -> ()
  }) : (i32) -> !test.i32
  // CHECK: return %[[RES]] : !test.i32
  return %0 : !test.i32
}

// The divisor is poison (0 - 1 with nuw), giving an empty range.
// This used to divide by zero in inferDivSRange.
// CHECK-LABEL: func.func @divs_by_poison_sub
// CHECK: index.divs
func.func @divs_by_poison_sub() -> index {
  %z = arith.constant 0 : index
  %o = arith.constant 1 : index
  %bad = arith.subi %z, %o overflow<nsw, nuw> : index
  %q = index.divs %o, %bad
  return %q : index
}
