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

// -------------------------------------------------------------------------- //
// round.
// -------------------------------------------------------------------------- //
func.func @func_roundf(%a : f32) {
  %r = math.round %a : f32
  vector.print %r : f32
  return
}

func.func @roundf() {
  // CHECK: 4
  %a = arith.constant 3.8 : f32
  call @func_roundf(%a) : (f32) -> ()

  // CHECK: -4
  %b = arith.constant -3.8 : f32
  call @func_roundf(%b) : (f32) -> ()

  // CHECK: 0
  %c = arith.constant 0.0 : f32
  call @func_roundf(%c) : (f32) -> ()

  // CHECK: -4
  %d = arith.constant -4.2 : f32
  call @func_roundf(%d) : (f32) -> ()

  // CHECK: -495
  %e = arith.constant -495.0 : f32
  call @func_roundf(%e) : (f32) -> ()

  // CHECK: 495
  %f = arith.constant 495.0 : f32
  call @func_roundf(%f) : (f32) -> ()

  // CHECK: 9
  %g = arith.constant 8.5 : f32
  call @func_roundf(%g) : (f32) -> ()

  // CHECK: -9
  %h = arith.constant -8.5 : f32
  call @func_roundf(%h) : (f32) -> ()

  return
}

// -------------------------------------------------------------------------- //
// pow.
// -------------------------------------------------------------------------- //
func.func @func_powff64(%a : f64, %b : f64) {
  %r = math.powf %a, %b : f64
  vector.print %r : f64
  return
}

func.func @powf() {
  // CHECK: 16
  %a   = arith.constant 4.0 : f64
  %a_p = arith.constant 2.0 : f64
  call @func_powff64(%a, %a_p) : (f64, f64) -> ()

  // CHECK: -nan
  %b   = arith.constant -3.0 : f64
  %b_p = arith.constant 3.0 : f64
  call @func_powff64(%b, %b_p) : (f64, f64) -> ()

  // CHECK: 2.343
  %c   = arith.constant 2.343 : f64
  %c_p = arith.constant 1.000 : f64
  call @func_powff64(%c, %c_p) : (f64, f64) -> ()

  // CHECK: 0.176171
  %d   = arith.constant 4.25 : f64
  %d_p = arith.constant -1.2  : f64
  call @func_powff64(%d, %d_p) : (f64, f64) -> ()

  // CHECK: 1
  %e   = arith.constant 4.385 : f64
  %e_p = arith.constant 0.00 : f64
  call @func_powff64(%e, %e_p) : (f64, f64) -> ()

  // CHECK: 6.62637
  %f    = arith.constant 4.835 : f64
  %f_p  = arith.constant 1.2 : f64
  call @func_powff64(%f, %f_p) : (f64, f64) -> ()

  // CHECK: -nan
  %g    = arith.constant 0xff80000000000000 : f64
  call @func_powff64(%g, %g) : (f64, f64) -> ()

  // CHECK: nan
  %h = arith.constant 0x7fffffffffffffff : f64
  call @func_powff64(%h, %h) : (f64, f64) -> ()

  // CHECK: nan
  %i = arith.constant 1.0 : f64
  call @func_powff64(%i, %h) : (f64, f64) -> ()

  // CHECK: inf
  %j   = arith.constant 29385.0 : f64
  %j_p = arith.constant 23598.0 : f64
  call @func_powff64(%j, %j_p) : (f64, f64) -> () 
  return
}

func.func @main() {
  call @exp2f() : () -> ()
  call @roundf() : () -> ()
  call @powf() : () -> ()
  return
}
