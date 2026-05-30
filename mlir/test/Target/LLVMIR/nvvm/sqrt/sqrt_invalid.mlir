// RUN: mlir-translate --mlir-to-llvmir --split-input-file --verify-diagnostics %s

// -----

llvm.func @sqrt_invalid_no_rnd(%a : f32) -> f32 {
  // expected-error@+1 {{rounding mode cannot be None}}
  %0 = nvvm.sqrt %a {rnd = #nvvm.fp_rnd_mode<none>} : f32
  llvm.return %0 : f32
}

// -----

llvm.func @sqrt_invalid_f64_ftz(%a : f64) -> f64 {
  // expected-error@+1 {{FTZ is not supported for f64}}
  %0 = nvvm.sqrt %a {rnd = #nvvm.fp_rnd_mode<rn>, ftz = true} : f64
  llvm.return %0 : f64
}
