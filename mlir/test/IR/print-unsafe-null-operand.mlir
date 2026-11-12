// RUN: mlir-opt %s --mlir-very-unsafe-disable-verifier-on-parsing 2>&1 | FileCheck %s
//
// Regression test for https://github.com/llvm/llvm-project/issues/182747
// Verify that printing does not crash when an operation has a null operand
// due to an unresolvable forward reference created with
// --mlir-very-unsafe-disable-verifier-on-parsing.

// CHECK: "scf.if"(<<NULL VALUE>>)
// CHECK: <<NULL TYPE>>
func.func @t() {
  scf.if %c {
    %c = arith.constant true
  }
  return
}