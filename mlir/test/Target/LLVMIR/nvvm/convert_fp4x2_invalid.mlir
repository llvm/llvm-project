// RUN: mlir-translate -mlir-to-llvmir -verify-diagnostics %s

// -----

llvm.func @convert_f16x2_to_f4x2_invalid_type(%src : vector<2xf16>) {
  // expected-error @below {{Only 'f4E2M1FN' type is supported for conversions from f16x2 to f4x2.}}
  %res = nvvm.convert.f16x2.to.f4x2 %src : vector<2xf16> -> i8 (f8E4M3FN)
  llvm.return
}

// -----

llvm.func @convert_bf16x2_to_f4x2_invalid_type(%src : vector<2xbf16>) {
  // expected-error @below {{Only 'f4E2M1FN' type is supported for conversions from bf16x2 to f4x2.}}
  %res = nvvm.convert.bf16x2.to.f4x2 %src : vector<2xbf16> -> i8 (f8E4M3FN)
  llvm.return
}
