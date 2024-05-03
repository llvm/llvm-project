// RUN: mlir-opt %s -allocate-arm-sme-tiles -convert-arm-sme-to-llvm -split-input-file -allow-unregistered-dialect -verify-diagnostics

//===----------------------------------------------------------------------===//
// arm_sme.outerproduct
//===----------------------------------------------------------------------===//

func.func @arm_sme_outerproduct_unsupported_type(%lhs : vector<[16]xi8>, %rhs : vector<[16]xi8>) {
  %acc = arm_sme.get_tile : vector<[16]x[16]xi8>
  // expected-error@+2 {{failed to legalize operation 'arm_sme.outerproduct'}}
  // expected-error@+1 {{unsupported type}}
  %0 = arm_sme.outerproduct %lhs, %rhs  acc(%acc) : vector<[16]xi8>, vector<[16]xi8>
  "prevent.dce"(%0) : (vector<[16]x[16]xi8>) -> ()
}

