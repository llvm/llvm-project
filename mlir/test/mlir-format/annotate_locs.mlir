// RUN: mlir-format %s --mlir-use-nameloc-as-prefix --insert-name-loc-only | FileCheck %s

// Append NameLocs (`loc("[ssa_name]")`) to operations and block arguments

// CHECK: func.func @add_one(%my_input: f64 loc("my_input"), %my_input2: f64 loc("my_input2")) -> f64 {
func.func @add_one(%my_input: f64, %my_input2: f64) -> f64 {
    // CHECK: %my_constant = arith.constant 1.00000e+00 : f64 loc("my_constant")
    %my_constant = arith.constant 1.00000e+00 : f64

    %my_output = arith.addf %my_input, %my_constant : f64
    // CHECK: %my_output = arith.addf %my_input, %my_constant : f64 loc("my_output")
    return %my_output : f64
}
