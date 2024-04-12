// RUN: mlir-opt %s --pass-pipeline='builtin.module(convert-math-to-libm{allow-c23-features=0 rounding-mode-is-default=0})' -verify-diagnostics

func.func @nearbyint_caller(%float: f32, %double: f64) -> (f32, f64)  {
  // expected-error@+1 {{failed to legalize operation 'math.roundeven'}}
  %float_result = math.roundeven %float : f32
  %double_result = math.roundeven %double : f64
  return %float_result, %double_result : f32, f64
}
