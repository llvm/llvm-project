// RUN: mlir-translate -mlir-to-llvmir -verify-diagnostics %s

// -----

llvm.func @convert_bf16x2_to_f8x2_invalid_type(%src : vector<2xbf16>) {
  // expected-error @below {{attribute 'dstTy' failed to satisfy constraint: type attribute of f8E8M0FNU type or f8E5M3FNU type or f8E4M3FN type or f8E5M2 type}}
  %res = nvvm.convert.bf16x2.to.f8x2 %src : vector<2xbf16> -> vector<2xi8> (f6E2M3FN)
  llvm.return
}

// -----

llvm.func @convert_bf16x2_to_f8x2_invalid_rounding_1(%src : vector<2xbf16>) {
  // expected-error @below {{Only RN and RZ rounding modes are supported for conversions from bf16x2 to 'f8E4M3FN' and 'f8E5M2' types}}
  %res = nvvm.convert.bf16x2.to.f8x2 %src rnd = <rm> : vector<2xbf16> -> vector<2xi8> (f8E4M3FN)
  llvm.return
}

// -----

llvm.func @nvvm_cvt_bf16x2_to_f8x2_invalid_rounding_2(%src : vector<2xbf16>) {
  // expected-error @below {{Only RZ and RP rounding modes are supported for conversions from bf16x2 to 'f8E8M0FNU' type}}
  %res = nvvm.convert.bf16x2.to.f8x2 %src rnd = <rn> : vector<2xbf16> -> vector<2xi8> (f8E8M0FNU)
  llvm.return
}

// -----

llvm.func @convert_bf16x2_to_f8x2_invalid_sat_mode(%src : vector<2xbf16>) {
  // expected-error @below {{Only SATFINITE saturation mode is supported for conversions from bf16x2 to 'f8E4M3FN' and 'f8E5M2' types}}
  %res = nvvm.convert.bf16x2.to.f8x2 %src rnd = <rn> sat = <none> : vector<2xbf16> -> vector<2xi8> (f8E4M3FN)
  llvm.return
}

// -----

llvm.func @convert_bf16x2_to_f8x2_invalid_relu(%src : vector<2xbf16>) {
  // expected-error @below {{relu not supported for conversions to 'f8E8M0FNU' type}}
  %res = nvvm.convert.bf16x2.to.f8x2 %src rnd = <rp> relu = true : vector<2xbf16> -> vector<2xi8> (f8E8M0FNU)
  llvm.return
}

// -----

llvm.func @convert_bf16x2_to_f8x2_invalid_pzo(%src : vector<2xbf16>) {
  // expected-error @below {{pzo not supported for conversions to 'f8E8M0FNU' type}}
  %res = nvvm.convert.bf16x2.to.f8x2 %src rnd = <rz> pzo = true : vector<2xbf16> -> vector<2xi8> (f8E8M0FNU)
  llvm.return
}

// -----

llvm.func @convert_f32x2_to_f8x2_scale_not_supported_ue8m0(%a : f32, %b : f32, %c : i8) {
  // expected-error @below {{scale not supported for conversions to 'f8E8M0FNU' type}}
  %res = nvvm.convert.f32x2.to.f8x2 %a, %b, %c rnd = <rp> : i16 (f8E8M0FNU)
  llvm.return
}

// -----

llvm.func @convert_bf16x2_to_f8x2_scale_not_supported_ue8m0(%src : vector<2xbf16>, %c : i8) {
  // expected-error @below {{scale not supported for conversions to 'f8E8M0FNU' type}}
  %res = nvvm.convert.bf16x2.to.f8x2 %src, %c rnd = <rp> : vector<2xbf16> -> i16 (f8E8M0FNU)
  llvm.return
}

// -----

llvm.func @convert_f16x2_to_f8x2_invalid_type(%src : vector<2xf16>) {
  // expected-error @below {{Only 'f8E5M3FNU', 'f8E4M3FN', and 'f8E5M2' types are supported for conversions from f16x2 to f8x2.}}
  %res = nvvm.convert.f16x2.to.f8x2 %src : vector<2xf16> -> i16 (f8E8M0FNU)
  llvm.return
}
