// RUN: mlir-opt %s --convert-arith-to-apfloat --convert-to-llvm | \
// RUN:   mlir-runner -e entry --entry-point-result=void \
// RUN:               --shared-libs=%mlir_c_runner_utils | FileCheck %s

// Put rhs into separate function so that it won't be constant-folded.
func.func @foo() -> f8E4M3FN {
  %cst = arith.constant 2.2 : f8E4M3FN
  return %cst : f8E4M3FN
}

func.func @entry() {
  %a = arith.constant 1.4 : f8E4M3FN
  %b = func.call @foo() : () -> (f8E4M3FN)
  %c = arith.addf %a, %b : f8E4M3FN

  // CHECK: 3.5
  vector.print %c : f8E4M3FN
  return
}
