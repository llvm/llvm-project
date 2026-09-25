// RUN: mlir-translate --mlir-to-llvmir --split-input-file --verify-diagnostics %s

// -----

llvm.func @fma_invalid_rnd_mode(%a : f16, %b : f16, %c : f16) -> f16 {
  // expected-error@+1 {{rounding mode must be specified}}
  %f1 = nvvm.fma %a, %b, %c, rnd = <none> : f16
  llvm.return %f1 : f16
}

// -----

llvm.func @fma_invalid_sat_mode(%a : f16, %b : f16, %c : f16) -> f16 {
  // expected-error@+1 {{attribute 'sat' failed to satisfy constraint: Describes the saturation mode whose value is one of {none, sat}}}
  %f1 = nvvm.fma %a, %b, %c, rnd = <rn> sat = <satfinite> : f16
  llvm.return %f1 : f16
}

// -----

llvm.func @fma_invalid_relu_sat(%a : f16, %b : f16, %c : f16) -> f16 {
  // expected-error@+1 {{relu and saturation are not supported together}}
  %f1 = nvvm.fma %a, %b, %c, rnd = <rn> sat = <sat> relu = true : f16
  llvm.return %f1 : f16
}

// -----

llvm.func @fma_invalid_oob_sat(%a : f16, %b : f16, %c : f16) -> f16 {
  // expected-error@+1 {{oob is not supported with saturation or FTZ}}
  %f1 = nvvm.fma %a, %b, %c, rnd = <rn> sat = <sat> oob = true : f16
  llvm.return %f1 : f16
}

// -----

llvm.func @fma_invalid_oob_f64(%a : f64, %b : f64, %c : f64) -> f64 {
  // expected-error@+1 {{relu and oob are only supported for f16 and bf16}}
  %f1 = nvvm.fma %a, %b, %c, rnd = <rn> oob = true : f64
  llvm.return %f1 : f64
}

// -----

llvm.func @fma_invalid_relu_oob(%a : f32, %b : f32, %c : f32) -> f32 {
  // expected-error@+1 {{relu and oob are only supported for f16 and bf16}}
  %f1 = nvvm.fma %a, %b, %c, rnd = <rn> relu = true : f32
  llvm.return %f1 : f32
}

// -----

llvm.func @fma_invalid_ftz_sat_f64(%a : f64, %b : f64, %c : f64) -> f64 {
  // expected-error@+1 {{FTZ and saturation are not supported for f64 type}}
  %f1 = nvvm.fma %a, %b, %c, rnd = <rn> sat = <sat> ftz = true : f64
  llvm.return %f1 : f64
}

// -----

llvm.func @fma_invalid_v2f16_rnd_mode(%a : vector<2xf16>, %b : vector<2xf16>, %c : vector<2xf16>) -> vector<2xf16> {
  // expected-error@+1 {{only RN rounding mode is supported for f16 and vector<2xf16>}}
  %f1 = nvvm.fma %a, %b, %c, rnd = <rm> : vector<2xf16>
  llvm.return %f1 : vector<2xf16>
}

// -----

llvm.func @fma_invalid_v2bf16_rnd_mode(%a : vector<2xbf16>, %b : vector<2xbf16>, %c : vector<2xbf16>) -> vector<2xbf16> {
  // expected-error@+1 {{only RN rounding mode is supported for bf16 and vector<2xbf16>}}
  %f1 = nvvm.fma %a, %b, %c, rnd = <rm> : vector<2xbf16>
  llvm.return %f1 : vector<2xbf16>
}

// -----

llvm.func @fma_invalid_ftz_v2bf16(%a : vector<2xbf16>, %b : vector<2xbf16>, %c : vector<2xbf16>) -> vector<2xbf16> {
  // expected-error@+1 {{FTZ and saturation are not supported for bf16 and vector<2xbf16>}}
  %f1 = nvvm.fma %a, %b, %c, rnd = <rn> ftz = true : vector<2xbf16>
  llvm.return %f1 : vector<2xbf16>
}

// -----

llvm.func @fma_invalid_sat_v2bf16(%a : vector<2xbf16>, %b : vector<2xbf16>, %c : vector<2xbf16>) -> vector<2xbf16> {
  // expected-error@+1 {{FTZ and saturation are not supported for bf16 and vector<2xbf16>}}
  %f1 = nvvm.fma %a, %b, %c, rnd = <rn> sat = <sat> : vector<2xbf16>
  llvm.return %f1 : vector<2xbf16>
}
