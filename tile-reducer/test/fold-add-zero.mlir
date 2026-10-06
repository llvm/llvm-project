// RUN: tr-opt %s --tr-fold-add-zero | FileCheck %s

// CHECK-LABEL: func.func @fold_rhs
func.func @fold_rhs(%x: !tr.tile<128xf32>) -> !tr.tile<128xf32> {
  %z = tr.constant 0.0 : !tr.tile<128xf32>
  // CHECK-NOT: tr.add
  // CHECK: return %[[X:.*]] : !tr.tile<128xf32>
  %r = tr.add %x, %z : !tr.tile<128xf32>
  return %r : !tr.tile<128xf32>
}

// CHECK-LABEL: func.func @fold_lhs
func.func @fold_lhs(%x: !tr.tile<128xf32>) -> !tr.tile<128xf32> {
  %z = tr.constant 0.0 : !tr.tile<128xf32>
  // CHECK-NOT: tr.add
  %r = tr.add %z, %x : !tr.tile<128xf32>
  return %r : !tr.tile<128xf32>
}

// CHECK-LABEL: func.func @no_fold
func.func @no_fold(%a: !tr.tile<128xf32>, %b: !tr.tile<128xf32>) -> !tr.tile<128xf32> {
  // CHECK: tr.add
  %r = tr.add %a, %b : !tr.tile<128xf32>
  return %r : !tr.tile<128xf32>
}
