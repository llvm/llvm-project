// RUN: mlir-opt %s -loop-invariant-code-motion -split-input-file | FileCheck %s

// Semi-affine map with an unproven divisor => must NOT hoist.
// CHECK-LABEL: func.func @no_hoist_semi_affine
// CHECK: scf.for
// CHECK:   affine.apply
func.func @no_hoist_semi_affine(%x: index, %c: index, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %j = %c0 to %n step %c1 {
    %q = affine.apply affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%x, %c]
    "test.use"(%q) : (index) -> ()
  }
  return
}

// -----

// Pure affine map => always speculatable, hoist is legal.
// CHECK-LABEL: func.func @hoist_pure_affine
// CHECK: affine.apply
// CHECK: scf.for
func.func @hoist_pure_affine(%x: index, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %j = %c0 to %n step %c1 {
    %q = affine.apply affine_map<(d0) -> (d0 floordiv 4)> (%x)
    "test.use"(%q) : (index) -> ()
  }
  return
}
