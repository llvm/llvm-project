// RUN: mlir-opt %s -loop-invariant-code-motion -split-input-file | FileCheck %s

// Dynamic basis with no proven positivity => must NOT hoist.
// CHECK-LABEL: func.func @no_hoist_dynamic_basis
// CHECK: scf.for
// CHECK:   affine.delinearize_index
func.func @no_hoist_dynamic_basis(%x: index, %c: index, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %j = %c0 to %n step %c1 {
    %q:2 = affine.delinearize_index %x into (%c) : index, index
    "test.use"(%q#0) : (index) -> ()
  }
  return
}

// -----

// Constant positive basis => hoist is legal.
// CHECK-LABEL: func.func @hoist_constant_basis
// CHECK: affine.delinearize_index
// CHECK: scf.for
func.func @hoist_constant_basis(%x: index, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %four = arith.constant 4 : index
  scf.for %j = %c0 to %n step %c1 {
    %q:2 = affine.delinearize_index %x into (%four) : index, index
    "test.use"(%q#0) : (index) -> ()
  }
  return
}
