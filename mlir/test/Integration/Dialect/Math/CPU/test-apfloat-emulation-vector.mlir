// REQUIRES: system-linux || system-darwin

// All floating-point arithmetics is lowered through APFloat.
// RUN: mlir-opt %s --convert-math-to-apfloat --convert-vector-to-scf \
// RUN:     --convert-scf-to-cf --convert-to-llvm | \
// RUN: mlir-runner -e entry --entry-point-result=void \
// RUN:             --shared-libs=%mlir_c_runner_utils \
// RUN:             --shared-libs=%mlir_apfloat_wrappers | FileCheck %s

func.func @entry() {

  %neg14fp8 = arith.constant dense<[-1.4, -1.4, -1.4, -1.4]> : vector<4xf8E4M3FN>
  %absfp8 = math.absf %neg14fp8 : vector<4xf8E4M3FN>
  // CHECK: ( 1.375, 1.375, 1.375, 1.375 )
  vector.print %absfp8 : vector<4xf8E4M3FN>

  %a1_vec = arith.constant dense<[2.0, 2.0, 2.0, 2.0]> : vector<4xf8E4M3FN>
  %b1_vec = arith.constant dense<[4.0, 4.0, 4.0, 4.0]> : vector<4xf8E4M3FN>
  %c1_vec = arith.constant dense<[8.0, 8.0, 8.0, 8.0]> : vector<4xf8E4M3FN>
  %d1_vec = math.fma %a1_vec, %b1_vec, %c1_vec : vector<4xf8E4M3FN>  // not supported by LLVM
  // CHECK: ( 16, 16, 16, 16 )
  vector.print %d1_vec : vector<4xf8E4M3FN>

  // CHECK: ( 0, 0, 0, 0 )
  %isinffp8 = math.isinf %neg14fp8 : vector<4xf8E4M3FN>
  vector.print %isinffp8 : vector<4xi1>

  %isnanfp8 = math.isnan %neg14fp8 : vector<4xf8E4M3FN>
  // CHECK: ( 0, 0, 0, 0 )
  vector.print %isnanfp8 : vector<4xi1>

  %isnormalfp8 = math.isnormal %neg14fp8 : vector<4xf8E4M3FN>
  // CHECK: ( 1, 1, 1, 1 )
  vector.print %isnormalfp8 : vector<4xi1>

  %isfinitefp8 = math.isfinite %neg14fp8 : vector<4xf8E4M3FN>
  // CHECK: ( 1, 1, 1, 1 )
  vector.print %isfinitefp8 : vector<4xi1>

  return
}
