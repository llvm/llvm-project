// RUN: mlir-translate --mlir-to-llvmir --split-input-file --verify-diagnostics %s

// -----

// Rounded form: rnd cannot be None.
llvm.func @divf_invalid_no_rnd(%a : f32, %b : f32) -> f32 {
  // expected-error@+1 {{rounding mode cannot be None for the rounded divide}}
  %0 = nvvm.divf %a, %b {rnd = #nvvm.fp_rnd_mode<none>} : f32
  llvm.return %0 : f32
}

// -----

// Rounded f64: FTZ not supported.
llvm.func @divf_invalid_f64_ftz(%a : f64, %b : f64) -> f64 {
  // expected-error@+1 {{FTZ is not supported for f64}}
  %0 = nvvm.divf %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, ftz = true} : f64
  llvm.return %0 : f64
}

// -----

// approx and full are mutually exclusive.
llvm.func @divf_invalid_approx_and_full(%a : f32, %b : f32) -> f32 {
  // expected-error@+1 {{'approx' and 'full' are mutually exclusive}}
  %0 = nvvm.divf %a, %b {approx = true, full = true} : f32
  llvm.return %0 : f32
}

// -----

// approx is f32-only.
llvm.func @divf_invalid_approx_f64(%a : f64, %b : f64) -> f64 {
  // expected-error@+1 {{'approx' and 'full' forms are f32-only}}
  %0 = nvvm.divf %a, %b {approx = true} : f64
  llvm.return %0 : f64
}

// -----

// full is f32-only.
llvm.func @divf_invalid_full_f64(%a : f64, %b : f64) -> f64 {
  // expected-error@+1 {{'approx' and 'full' forms are f32-only}}
  %0 = nvvm.divf %a, %b {full = true} : f64
  llvm.return %0 : f64
}

// -----

// approx does not accept a rounding mode.
llvm.func @divf_invalid_approx_with_rnd(%a : f32, %b : f32) -> f32 {
  // expected-error@+1 {{'approx' and 'full' forms do not accept a rounding mode}}
  %0 = nvvm.divf %a, %b {approx = true, rnd = #nvvm.fp_rnd_mode<rn>} : f32
  llvm.return %0 : f32
}

// -----

// full does not accept a rounding mode.
llvm.func @divf_invalid_full_with_rnd(%a : f32, %b : f32) -> f32 {
  // expected-error@+1 {{'approx' and 'full' forms do not accept a rounding mode}}
  %0 = nvvm.divf %a, %b {full = true, rnd = #nvvm.fp_rnd_mode<rn>} : f32
  llvm.return %0 : f32
}
