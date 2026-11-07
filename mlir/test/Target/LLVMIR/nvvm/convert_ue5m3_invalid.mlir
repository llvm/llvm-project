// RUN: mlir-translate -mlir-to-llvmir -verify-diagnostics %s

// -----

llvm.func @convert_f32x2_to_f8x2_invalid_rounding_ue5m3(%a : f32, %b : f32) {
  // expected-error @below {{Only RN, RZ, and RP rounding modes are supported for conversions from f32x2 to 'f8E5M3FNU' type}}
  %res = nvvm.convert.f32x2.to.f8x2 %a, %b rnd = <rm> : i16 (f8E5M3FNU)
  llvm.return
}

// -----

llvm.func @convert_f32x2_to_f8x2_invalid_relu_ue5m3(%a : f32, %b : f32) {
  // expected-error @below {{relu not supported for conversions to 'f8E5M3FNU' type}}
  %res = nvvm.convert.f32x2.to.f8x2 %a, %b rnd = <rn> relu = true : i16 (f8E5M3FNU)
  llvm.return
}

// -----

llvm.func @convert_f32x2_to_f8x2_invalid_pzo_ue5m3(%a : f32, %b : f32) {
  // expected-error @below {{pzo not supported for conversions to 'f8E5M3FNU' type}}
  %res = nvvm.convert.f32x2.to.f8x2 %a, %b pzo = true rnd = <rz> : i16 (f8E5M3FNU)
  llvm.return
}

// -----

llvm.func @convert_f32x2_to_f8x2_invalid_scale_rp_ue5m3(%a : f32, %b : f32, %c : i8) {
  // expected-error @below {{scale not supported for conversions to 'f8E5M3FNU' type with RP rounding mode}}
  %res = nvvm.convert.f32x2.to.f8x2 %a, %b, %c rnd = <rp> : i16 (f8E5M3FNU)
  llvm.return
}

// -----

llvm.func @convert_f16x2_to_f8x2_invalid_rounding_ue5m3(%src : vector<2xf16>) {
  // expected-error @below {{Only RN, RZ, and RP rounding modes are supported for conversions from f16x2 to 'f8E5M3FNU' type}}
  %res = nvvm.convert.f16x2.to.f8x2 %src rnd = <rm> : vector<2xf16> -> i16 (f8E5M3FNU)
  llvm.return
}

// -----

llvm.func @convert_f16x2_to_f8x2_invalid_relu_ue5m3(%src : vector<2xf16>) {
  // expected-error @below {{relu not supported for conversions to 'f8E5M3FNU' type}}
  %res = nvvm.convert.f16x2.to.f8x2 %src rnd = <rp> relu = true : vector<2xf16> -> i16 (f8E5M3FNU)
  llvm.return
}

// -----

llvm.func @convert_f16x2_to_f8x2_invalid_pzo_ue5m3(%src : vector<2xf16>) {
  // expected-error @below {{pzo not supported for conversions to 'f8E5M3FNU' type}}
  %res = nvvm.convert.f16x2.to.f8x2 %src pzo = true rnd = <rz> : vector<2xf16> -> i16 (f8E5M3FNU)
  llvm.return
}

// -----

llvm.func @convert_f16x2_to_f8x2_invalid_scale_rp_ue5m3(%src : vector<2xf16>, %c : i8) {
  // expected-error @below {{scale not supported for conversions to 'f8E5M3FNU' type with RP rounding mode}}
  %res = nvvm.convert.f16x2.to.f8x2 %src, %c rnd = <rp> : vector<2xf16> -> i16 (f8E5M3FNU)
  llvm.return
}

// -----

llvm.func @convert_bf16x2_to_f8x2_invalid_rounding_ue5m3(%src : vector<2xbf16>) {
  // expected-error @below {{Only RN, RZ, and RP rounding modes are supported for conversions from bf16x2 to 'f8E5M3FNU' type}}
  %res = nvvm.convert.bf16x2.to.f8x2 %src rnd = <rm> : vector<2xbf16> -> i16 (f8E5M3FNU)
  llvm.return
}

// -----

llvm.func @convert_bf16x2_to_f8x2_invalid_relu_ue5m3(%src : vector<2xbf16>) {
  // expected-error @below {{relu not supported for conversions to 'f8E5M3FNU' type}}
  %res = nvvm.convert.bf16x2.to.f8x2 %src rnd = <rp> relu = true : vector<2xbf16> -> i16 (f8E5M3FNU)
  llvm.return
}

// -----

llvm.func @convert_bf16x2_to_f8x2_invalid_pzo_ue5m3(%src : vector<2xbf16>) {
  // expected-error @below {{pzo not supported for conversions to 'f8E5M3FNU' type}}
  %res = nvvm.convert.bf16x2.to.f8x2 %src pzo = true rnd = <rz> : vector<2xbf16> -> i16 (f8E5M3FNU)
  llvm.return
}

// -----

llvm.func @convert_bf16x2_to_f8x2_invalid_scale_rp_ue5m3(%src : vector<2xbf16>, %c : i8) {
  // expected-error @below {{scale not supported for conversions to 'f8E5M3FNU' type with RP rounding mode}}
  %res = nvvm.convert.bf16x2.to.f8x2 %src, %c rnd = <rp> : vector<2xbf16> -> i16 (f8E5M3FNU)
  llvm.return
}

// -----

llvm.func @convert_ue5m3x2_to_f16x2_invalid_relu(%src : vector<2xi8>) {
  // expected-error @below {{relu not supported for conversions from 'f8E5M3FNU' type.}}
  %res = nvvm.convert.f8x2.to.f16x2 %src relu = true : vector<2xi8> (f8E5M3FNU) -> vector<2xf16>
  llvm.return
}

// -----

llvm.func @convert_ue5m3x2_to_bf16x2_invalid_relu(%src : vector<2xi8>) {
  // expected-error @below {{relu not supported for conversions from 'f8E5M3FNU' type}}
  %res = nvvm.convert.f8x2.to.bf16x2 %src relu = true : vector<2xi8> (f8E5M3FNU) -> vector<2xbf16>
  llvm.return
}
