// RUN: mlir-opt -retain-identifier-names %s | FileCheck %s


//===----------------------------------------------------------------------===//
// Test SSA results (with single return values)
//===----------------------------------------------------------------------===//

// CHECK: func.func @add_one(%my_input: f64) -> f64 {
func.func @add_one(%my_input: f64) -> f64 {
  // CHECK: %my_constant = arith.constant 1.000000e+00 : f64
  %my_constant = arith.constant 1.000000e+00 : f64
  // CHECK: %my_output = arith.addf %my_input, %my_constant : f64
  %my_output = arith.addf %my_input, %my_constant : f64
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

//===----------------------------------------------------------------------===//
// Test multiple return values
//===----------------------------------------------------------------------===//

func.func @select_min_max(%a: f64, %b: f64) -> (f64, f64) {
  %gt = arith.cmpf "ogt", %a, %b : f64
  // CHECK: %min, %max = scf.if %gt -> (f64, f64) {
  %min, %max = scf.if %gt -> (f64, f64) {
    scf.yield %b, %a : f64, f64
  } else {
    scf.yield %a, %b : f64, f64
  }
  // CHECK: return %min, %max : f64, f64
  return %min, %max : f64, f64
}

// -----

//===----------------------------------------------------------------------===//
// Test multiple return values, with a grouped value tuple
//===----------------------------------------------------------------------===//

func.func @select_max(%a: f64, %b: f64, %c: f64, %d: f64) -> (f64, f64, f64, f64) {
  // Find the max between %a and %b,
  // with %c and %d being other values that are returned.
  %gt = arith.cmpf "ogt", %a, %b : f64
  // CHECK: %max, %others:2, %alt = scf.if %gt -> (f64, f64, f64, f64) {
  %max, %others:2, %alt  = scf.if %gt -> (f64, f64, f64, f64) {
    scf.yield %b, %a, %c, %d : f64, f64, f64, f64
  } else {
    scf.yield %a, %b, %d, %c : f64, f64, f64, f64
  }
  // CHECK: return %max, %others#0, %others#1, %alt : f64, f64, f64, f64
  return %max, %others#0, %others#1, %alt : f64, f64, f64, f64
}

// -----

//===----------------------------------------------------------------------===//
// Test identifiers which may clash with OpAsmOpInterface names (e.g., cst, %1, etc)
//===----------------------------------------------------------------------===//

// CHECK: func.func @clash(%arg1: f64, %arg0: f64, %arg2: f64) -> f64 {
func.func @clash(%arg1: f64, %arg0: f64, %arg2: f64) -> f64 {
  %my_constant = arith.constant 1.000000e+00 : f64
  // CHECK: %cst = arith.constant 2.000000e+00 : f64
  %cst = arith.constant 2.000000e+00 : f64
  // CHECK: %cst_1 = arith.constant 3.000000e+00 : f64
  %cst_1 = arith.constant 3.000000e+00 : f64
  // CHECK: %1 = arith.addf %arg1, %cst : f64
  %1 = arith.addf %arg1, %cst : f64
  // CHECK: %0 = arith.addf %arg1, %cst_1 : f64
  %0 = arith.addf %arg1, %cst_1 : f64
  // CHECK: return %1 : f64
  return %1 : f64
}

// -----
