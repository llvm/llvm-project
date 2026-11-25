// RUN: mlir-translate -mlir-to-llvmir -verify-diagnostics %s

// -----

llvm.func @convert_f16x2_to_f4x2_invalid_type(%src : vector<2xf16>) {
  // expected-error @below {{attribute 'dstTy' failed to satisfy constraint: type attribute of f4E2M1FN type}}
  %res = nvvm.convert.f16x2.to.f4x2 %src : vector<2xf16> -> i8 (f8E4M3FN)
  llvm.return
}

// -----

llvm.func @convert_bf16x2_to_f4x2_invalid_type(%src : vector<2xbf16>) {
  // expected-error @below {{attribute 'dstTy' failed to satisfy constraint: type attribute of f4E2M1FN type}}
  %res = nvvm.convert.bf16x2.to.f4x2 %src : vector<2xbf16> -> i8 (f8E4M3FN)
  llvm.return
}

// -----

llvm.func @convert_f32x2_to_f4x2_invalid_rounding(%a : f32, %b : f32) {
  // expected-error @below {{attribute 'rnd' failed to satisfy constraint: NVVM FPRoundingMode kind whose value is one of {rn, rz}}}
  %res = nvvm.convert.f32x2.to.f4x2 %a, %b rnd = <rp> : i8 (f4E2M1FN)
  llvm.return
}

// -----

llvm.func @convert_bf16x2_to_f4x2_invalid_rounding(%src : vector<2xbf16>) {
  // expected-error @below {{attribute 'rnd' failed to satisfy constraint: NVVM FPRoundingMode kind whose value is one of {rn, rz}}}
  %res = nvvm.convert.bf16x2.to.f4x2 %src rnd = <rs> : vector<2xbf16> -> i8 (f4E2M1FN)
  llvm.return
}
