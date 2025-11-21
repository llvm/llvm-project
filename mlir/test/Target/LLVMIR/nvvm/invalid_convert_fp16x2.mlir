// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

llvm.func @nvvm_cvt_f32x2_to_f16x2_invalid_rounding(%a : f32, %b : f32) {
  // expected-error @below {{Only RN, RZ, and RS rounding modes are supported for conversions from f32x2 to f16x2.}}
  %res = nvvm.convert.f32x2.to.f16x2 %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xf16>
  llvm.return
}

// -----

llvm.func @nvvm_cvt_f32x2_to_f16x2_invalid_rbits_1(%a : f32, %b : f32) {
  // expected-error @below {{random_bits is required for RS rounding mode.}}
  %res = nvvm.convert.f32x2.to.f16x2 %a, %b {rnd = #nvvm.fp_rnd_mode<rs>} : vector<2xf16>
  llvm.return
}

// -----

llvm.func @nvvm_cvt_f32x2_to_f16x2_invalid_rbits_2(%a : f32, %b : f32, %rbits : i32) {
  // expected-error @below {{random_bits not supported for RN and RZ rounding modes.}}
  %res = nvvm.convert.f32x2.to.f16x2 %a, %b, %rbits {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xf16>
  llvm.return
}

// -----

llvm.func @nvvm_cvt_f32x2_to_bf16x2_invalid_rounding(%a : f32, %b : f32) {
  // expected-error @below {{Only RN, RZ, and RS rounding modes are supported for conversions from f32x2 to bf16x2.}}
  %res = nvvm.convert.f32x2.to.bf16x2 %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xbf16>
  llvm.return
}

// -----

llvm.func @nvvm_cvt_f32x2_to_bf16x2_invalid_rbits_1(%a : f32, %b : f32) {
  // expected-error @below {{random_bits is required for RS rounding mode.}}
  %res = nvvm.convert.f32x2.to.bf16x2 %a, %b {rnd = #nvvm.fp_rnd_mode<rs>} : vector<2xbf16>
  llvm.return
}

llvm.func @nvvm_cvt_f32x2_to_bf16x2_invalid_rbits_2(%a : f32, %b : f32, %rbits : i32) {
  // expected-error @below {{random_bits not supported for RN and RZ rounding modes.}}
  %res = nvvm.convert.f32x2.to.bf16x2 %a, %b, %rbits {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xbf16>
  llvm.return
}
