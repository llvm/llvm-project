// RUN: mlir-format %s --mlir-use-nameloc-as-prefix  | FileCheck %s

// CHECK: func.func @add_one(%my_input: f64) -> f64 {
func.func @add_one(%my_input: f64) -> f64 {
    // CHECK: %my_constant = arith.constant 1.00000e+00 : f64
    %my_constant = arith.constant 1.00000e+00 : f64
    // CHECK: // Dinnae drop this comment!
    // Dinnae drop this comment!
    %my_output = arith.addf
               %my_input,
               %my_constant : f64
    // CHECK-STRICT: %my_output = arith.addf %my_input, %my_constant : f64
    return %my_output : f64
    // CHECK: return %my_output : f64
}
