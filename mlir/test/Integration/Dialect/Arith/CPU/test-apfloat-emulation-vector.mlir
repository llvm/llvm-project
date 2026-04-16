// REQUIRES: system-linux || system-darwin

// All floating-point arithmetics is lowered through APFloat.
// RUN: mlir-opt %s --convert-arith-to-apfloat --convert-vector-to-scf \
// RUN:     --convert-scf-to-cf --convert-to-llvm | \
// RUN: mlir-runner -e entry --entry-point-result=void \
// RUN:             --shared-libs=%mlir_c_runner_utils \
// RUN:             --shared-libs=%mlir_apfloat_wrappers | FileCheck %s

// Put rhs into separate function so that it won't be constant-folded.
func.func @foo_vec() -> (vector<4xf8E4M3FN>, vector<4xf32>) {
  %cst1 = arith.constant dense<[2.2, 2.2, 2.2, 2.2]> : vector<4xf8E4M3FN>
  %cst2 = arith.constant dense<[2.2, 2.2, 2.2, 2.2]> : vector<4xf32>
  return %cst1, %cst2 : vector<4xf8E4M3FN>, vector<4xf32>
}

func.func @entry() {
  // CHECK: ( 3.5, 3.5, 3.5, 3.5 )
  %a1_vec = arith.constant dense<[1.4, 1.4, 1.4, 1.4]> : vector<4xf8E4M3FN>
  %b1_vec, %b2_vec = func.call @foo_vec() : () -> (vector<4xf8E4M3FN>, vector<4xf32>)
  %c1_vec = arith.addf %a1_vec, %b1_vec : vector<4xf8E4M3FN>  // not supported by LLVM
  vector.print %c1_vec : vector<4xf8E4M3FN>
  return
}
