// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// `llvm.fneg` has no constrained floating-point intrinsic.
llvm.func @fneg(%a: f32) -> f32 {
  // expected-error @below {{no constrained intrinsic is available for 'llvm.fneg' carrying a 'fenv' attribute}}
  // expected-error @below {{LLVM Translation failed for operation: llvm.fneg}}
  %0 = llvm.fneg %a {fenv = #llvm.fenv<rounding_mode = upward>} : f32
  llvm.return %0 : f32
}

// -----

// A named intrinsic without a constrained counterpart is rejected.
llvm.func @call_intrinsic(%a: f32) -> f32 {
  // expected-error @below {{no constrained intrinsic is available for 'llvm.fabs.f32' carrying a 'fenv' attribute}}
  // expected-error @below {{LLVM Translation failed for operation: llvm.call_intrinsic}}
  %0 = llvm.call_intrinsic "llvm.fabs.f32"(%a) {fenv = #llvm.fenv<rounding_mode = upward>} : (f32) -> f32
  llvm.return %0 : f32
}
