// RUN: mlir-opt -convert-scf-to-emitc %s -split-input-file -verify-diagnostics

func.func @unsupported_type_vector(%arg0 : index, %arg1 : index, %arg2 : index) -> vector<3xindex> {
  %zero = arith.constant dense<0> : vector<3xindex>
  // expected-error@+1 {{failed to legalize operation 'scf.for'}}
  %r = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%acc = %zero) -> vector<3xindex> {
    scf.yield %acc : vector<3xindex>
  }
  return %r : vector<3xindex>
}
