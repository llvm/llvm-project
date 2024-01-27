// RUN: mlir-opt -retain-identifier-names %s | FileCheck %s


//===----------------------------------------------------------------------===//
// Test SSA results (with single return values)
//===----------------------------------------------------------------------===//

// CHECK: func.func @add_one(%arg0: f64, %arg1: f64) -> f64 {
func.func @add_one(%arg0: f64, %arg1: f64) -> f64 {
  // CHECK: %my_constant = arith.constant 1.000000e+00 : f64
  %my_constant = arith.constant 1.000000e+00 : f64
  // CHECK: %my_output = arith.addf %arg0, %my_constant : f64
  %my_output = arith.addf %arg0, %my_constant : f64
  // CHECK: return %my_output : f64
  return %my_output : f64
}


// -----

//===----------------------------------------------------------------------===//
// Test basic blocks and their arguments
//===----------------------------------------------------------------------===//

func.func @simple(i64, i1) -> i64 {
^bb_alpha(%a: i64, %cond: i1):
  // CHECK: cf.cond_br %cond, ^bb_beta, ^bb_gamma
  cf.cond_br %cond, ^bb_beta, ^bb_gamma

// CHECK: ^bb_beta:  // pred: ^bb_alpha
^bb_beta:
  // CHECK: cf.br ^bb_delta(%a : i64)
  cf.br ^bb_delta(%a: i64)

// CHECK: ^bb_gamma:  // pred: ^bb_alpha
^bb_gamma:
  // CHECK: %b = arith.addi %a, %a : i64
  %b = arith.addi %a, %a : i64
  // CHECK: cf.br ^bb_delta(%b : i64)
  cf.br ^bb_delta(%b: i64)

// CHECK: ^bb_delta(%c: i64):  // 2 preds: ^bb_gamma, ^bb_beta
^bb_delta(%c: i64):
  // CHECK: cf.br ^bb_eps(%c, %a : i64, i64)
  cf.br ^bb_eps(%c, %a : i64, i64)

// CHECK: ^bb_eps(%d: i64, %e: i64):  // pred: ^bb_delta
^bb_eps(%d : i64, %e : i64):
  // CHECK: %f = arith.addi %d, %e : i64
  %f = arith.addi %d, %e : i64
  return %f : i64
}

// -----
