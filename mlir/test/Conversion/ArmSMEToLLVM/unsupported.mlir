// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(convert-arm-sme-to-llvm))" -verify-diagnostics -split-input-file

//===----------------------------------------------------------------------===//
// arm_sme.outerproduct
//===----------------------------------------------------------------------===//

func.func @arm_sme_outerproduct_unsupported_type(%lhs : vector<[16]xi8>, %rhs : vector<[16]xi8>) {
  %acc = arm_sme.get_tile : vector<[16]x[16]xi8>
  // expected-error@below {{unexpected operation with SME tile type after conversion to LLVM}}
  // expected-error@+2 {{failed to legalize operation 'arm_sme.outerproduct'}}
  // expected-error@+1 {{unsupported type}}
  %0 = arm_sme.outerproduct %lhs, %rhs  acc(%acc) : vector<[16]xi8>, vector<[16]xi8>
  "test.some_use"(%0) : (vector<[16]x[16]xi8>) -> ()
}

//===----------------------------------------------------------------------===//
// Unsupported operations on SME tile types
//===----------------------------------------------------------------------===//

// -----

func.func @unsupported_arith_op(%a : vector<[4]x[4]xf32>, %b : vector<[4]x[4]xf32>) {
  // expected-error@below {{unexpected operation with SME tile type after conversion to LLVM}}
  %0 = arith.addf %a, %b : vector<[4]x[4]xf32>
  "test.some_use"(%0) : (vector<[4]x[4]xf32>) -> ()
}
