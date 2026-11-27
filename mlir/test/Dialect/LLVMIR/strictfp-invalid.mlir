// RUN: mlir-opt %s -split-input-file -verify-diagnostics

llvm.func @strictfp_not_unit(%arg0: f32) -> f32 {
  // expected-error @below{{expected 'llvm.strictfp' to be a unit attribute}}
  %0 = llvm.intr.fabs(%arg0) {llvm.strictfp = 1 : i32} : (f32) -> f32
  llvm.return %0 : f32
}

// -----

llvm.func @f()

llvm.func @strictfp_on_call() {
  // expected-error @below{{expected 'llvm.strictfp' on an intrinsic op}}
  llvm.call @f() {llvm.strictfp} : () -> ()
  llvm.return
}
