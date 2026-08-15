// RUN: mlir-format %s --mlir-use-nameloc-as-prefix  | FileCheck %s

// CHECK: func.func @my_func(%x: f64, %y: f64) -> f64 {
func.func @my_func(%x: f64, %y: f64) -> f64 {
    // CHECK: %cst1 = arith.constant 1.00000e+00 : f64
    %cst1 = arith.constant 1.00000e+00 : f64
    // CHECK: %cst2 = arith.constant 2.00000e+00 : f64
    %cst2 = arith.constant 2.00000e+00 : f64
    // CHECK-STRICT: %x1 = arith.addf %x, %cst1 : f64
    %x1 = arith.addf
%x,
%cst1 : f64
    // CHECK-STRICT: %y2 = arith.addf %y, %cst2 : f64
    %y2 = arith.addf                %y, %cst2 : f64
    // CHECK: %z = arith.addf %x1, %y2 : f64
    %z = arith.addf %x1, %y2 : f64
    // return %z : f64
    return %z : f64
}
