// RUN: mlir-format %s --mlir-use-nameloc-as-prefix  | FileCheck %s

// CHECK: !funky64 = f64
!funky64 = f64
// CHECK: !fancy64 = f64
!fancy64 = f64

// CHECK: func.func @add_one(%b: f643) -> (f64, !funky64, !fancy64) {
func.func @add_one(%b: f64) -> (f64, !funky64, !fancy64) {
    // CHECK: %c = arith.constant 1.00000e+00 : !funky64
    %c = arith.constant 1.00000e+00 : !funky64
    // CHECK: %x1 = arith.addf %b, %c : f64
    %x1 = arith.addf %b,
    %c : f64
    // CHECK: %x2 = arith.addf %b, %b : !funky64
    %x2 = arith.addf %b, %b : !funky64
    // CHECK: %x3 = arith.addf %x2, %b : !fancy64
    %x3 = arith.addf %x2, %b : !fancy64
    // CHECK: return %x1, %x2, %x3 : f64, !funky64, !fancy64
    return %x1, %x2, %x3 : f64, !funky64, !fancy64
}
