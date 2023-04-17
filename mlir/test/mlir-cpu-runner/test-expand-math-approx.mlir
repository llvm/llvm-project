// RUN:   mlir-opt %s -pass-pipeline="builtin.module(func.func(test-expand-math,convert-arith-to-llvm),convert-vector-to-llvm,func.func(convert-math-to-llvm),convert-func-to-llvm,reconcile-unrealized-casts)" \
// RUN: | mlir-cpu-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%mlir_c_runner_utils  \
// RUN:     -shared-libs=%mlir_runner_utils    \
// RUN: | FileCheck %s

// -------------------------------------------------------------------------- //
// exp2f.
// -------------------------------------------------------------------------- //
func.func @func_exp2f(%a : f64) {
  %r = math.exp2 %a : f64
  vector.print %r : f64
  return
}

func.func @exp2f() {
  // CHECK: 2
  %a = arith.constant 1.0 : f64
  call @func_exp2f(%a) : (f64) -> ()  

  // CHECK: 4
  %b = arith.constant 2.0 : f64
  call @func_exp2f(%b) : (f64) -> ()

  // CHECK: 5.65685
  %c = arith.constant 2.5 : f64
  call @func_exp2f(%c) : (f64) -> ()

  // CHECK: 0.29730
  %d = arith.constant -1.75 : f64
  call @func_exp2f(%d) : (f64) -> ()

  // CHECK: 1.09581
  %e = arith.constant 0.132 : f64
  call @func_exp2f(%e) : (f64) -> ()

  // CHECK: inf
  %f1 = arith.constant 0.00 : f64
  %f2 = arith.constant 1.00 : f64
  %f = arith.divf %f2, %f1 : f64
  call @func_exp2f(%f) : (f64) -> ()

  // CHECK: inf
  %g = arith.constant 5038939.0 : f64
  call @func_exp2f(%g) : (f64) -> ()

  // CHECK: 0
  %neg_inf = arith.constant 0xff80000000000000 : f64
  call @func_exp2f(%neg_inf) : (f64) -> ()

  // CHECK: inf
  %i = arith.constant 0x7fc0000000000000 : f64
  call @func_exp2f(%i) : (f64) -> ()
  return
}

func.func @main() {
  call @exp2f() : () -> ()
  return
}
