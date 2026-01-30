// RUN: mlir-translate --mlir-to-llvmir --split-input-file --verify-diagnostics %s

// -----

llvm.func @fadd_invalid_sat_mode(%a : f16, %b : f16) -> f32 {
  // expected-error@+1 {{SATFINITE saturation mode is not supported for floating point addition operation}}
  %f1 = nvvm.fadd %a, %b {sat = #nvvm.sat_mode<satfinite>} : f16, f16 -> f32
  llvm.return %f1 : f32
}

// -----

llvm.func @fadd_invalid_vector_scalar_mix(%a : vector<2xf16>, %b : f16) -> f32 {
  // expected-error@+1 {{cannot mix vector and scalar types for floating point addition operation}}
  %f1 = nvvm.fadd %a, %b : vector<2xf16>, f16 -> f32
  llvm.return %f1 : f32
}

// -----

llvm.func @fadd_invalid_vector_element_types_mix(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xbf16> {
  // expected-error@+1 {{cannot mix different element types for vector floating point addition operation}}
  %f1 = nvvm.fadd %a, %b : vector<2xf16>, vector<2xbf16> -> vector<2xbf16>
  llvm.return %f1 : vector<2xbf16>
}

// -----

llvm.func @fadd_invalid_f64_sat_ftz(%a : f64, %b : f64) -> f64 {
  // expected-error@+1 {{FTZ and saturation are not supported for additions involving f64 type}}
  %f1 = nvvm.fadd %a, %b {sat = #nvvm.sat_mode<sat>, ftz} : f64, f64 -> f64
  llvm.return %f1 : f64
}

// -----

llvm.func @fadd_invalid_result_width(%a : f64, %b : f64) -> f32 {
  // expected-error@+1 {{result type must be at least as wide as the operands}}
  %f1 = nvvm.fadd %a, %b : f64, f64 -> f32
  llvm.return %f1 : f32
}

// ----

llvm.func @fadd_invalid_f16_rnd_mode(%a : f16, %b : f16) -> f16 {
  // expected-error@+1 {{only RN rounding mode is supported for f16 and vector<2xf16> additions}}
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : f16, f16 -> f16
  llvm.return %f1 : f16
}

// -----

llvm.func @fadd_invalid_v2f16_rnd_mode(%a : vector<2xf16>, %b : vector<2xf16>) -> vector<2xf16> {
  // expected-error@+1 {{only RN rounding mode is supported for f16 and vector<2xf16> additions}}
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xf16>, vector<2xf16> -> vector<2xf16>
  llvm.return %f1 : vector<2xf16>
}

// -----

llvm.func @fadd_invalid_bf16_rnd_mode(%a : bf16, %b : bf16) -> bf16 {
  // expected-error@+1 {{only RN rounding mode is supported for bf16 and vector<2xbf16> additions}}
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : bf16, bf16 -> bf16
  llvm.return %f1 : bf16
}

// -----

llvm.func @fadd_invalid_v2bf16_rnd_mode(%a : vector<2xbf16>, %b : vector<2xbf16>) -> vector<2xbf16> {
  // expected-error@+1 {{only RN rounding mode is supported for bf16 and vector<2xbf16> additions}}
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xbf16>, vector<2xbf16> -> vector<2xbf16>
  llvm.return %f1 : vector<2xbf16>
}

// -----

llvm.func @fadd_invalid_bf16_sat_ftz(%a : bf16, %b : bf16) -> bf16 {
  // expected-error@+1 {{FTZ and saturation are not supported for bf16 and vector<2xbf16> additions}}
  %f1 = nvvm.fadd %a, %b {sat = #nvvm.sat_mode<sat>, ftz} : bf16, bf16 -> bf16
  llvm.return %f1 : bf16
}

// -----

llvm.func @fadd_invalid_f16_result_type(%a : f16, %b : bf16) -> f16 {
  // expected-error@+1 {{only f16 + f16 is supported for f16 result type}}
  %f1 = nvvm.fadd %a, %b : f16, bf16 -> f16
  llvm.return %f1 : f16
}

// -----

llvm.func @fadd_invalid_bf16_result_type(%a : bf16, %b : f16) -> bf16 {
  // expected-error@+1 {{only bf16 + bf16 is supported for bf16 result type}}
  %f1 = nvvm.fadd %a, %b : bf16, f16 -> bf16
  llvm.return %f1 : bf16
}

// -----

// FIXME: Remove this test once intrinsics for f16 addition (with FTZ only) are 
// available.
llvm.func @fadd_invalid_f16_ftz_no_sat(%a : f16, %b : f16) -> f16 {
  // expected-error@+1 {{FTZ with no saturation is not supported for f16 additions}}
  %f1 = nvvm.fadd %a, %b {ftz} : f16, f16 -> f16
  llvm.return %f1 : f16
}
