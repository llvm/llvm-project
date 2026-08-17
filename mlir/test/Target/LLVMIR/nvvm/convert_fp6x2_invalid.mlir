// RUN: mlir-translate -mlir-to-llvmir -verify-diagnostics %s

// -----

llvm.func @convert_f16x2_to_f6x2_invalid_type(%src : vector<2xf16>) {
  // expected-error @below {{attribute 'dstTy' failed to satisfy constraint: type attribute of f6E2M3FN type or f6E3M2FN type}}
  %res = nvvm.convert.f16x2.to.f6x2 %src : vector<2xf16> -> vector<2xi8> (f8E4M3FN)
  llvm.return
}

// -----

llvm.func @convert_bf16x2_to_f6x2_invalid_type(%src : vector<2xbf16>) {
  // expected-error @below {{attribute 'dstTy' failed to satisfy constraint: type attribute of f6E2M3FN type or f6E3M2FN type}}
  %res = nvvm.convert.bf16x2.to.f6x2 %src : vector<2xbf16> -> vector<2xi8> (f8E4M3FN)
  llvm.return
}
