// RUN: mlir-translate --mlir-to-llvmir --split-input-file --verify-diagnostics %s

// -----

llvm.func @divf_invalid_no_rnd(%a : f32, %b : f32) -> f32 {
  // expected-error@+1 {{rounding mode cannot be None}}
  %0 = nvvm.divf %a, %b {rnd = #nvvm.fp_rnd_mode<none>} : f32
  llvm.return %0 : f32
}

// -----

llvm.func @divf_invalid_f64_ftz(%a : f64, %b : f64) -> f64 {
  // expected-error@+1 {{FTZ is not supported for f64}}
  %0 = nvvm.divf %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, ftz = true} : f64
  llvm.return %0 : f64
}
