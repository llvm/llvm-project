// RUN: mlir-translate --mlir-to-llvmir --split-input-file --verify-diagnostics %s

// -----

llvm.func @ex2_invalid_ftz_f16x2(%arg0: vector<2xf16>) -> vector<2xf16> {
  // expected-error@+1 {{FTZ is not supported for vector<2xf16>}}
  %0 = nvvm.ex2 %arg0 ftz = true : vector<2xf16>
  llvm.return %0 : vector<2xf16>
}

// -----

llvm.func @ex2_invalid_no_ftz_bf16x2(%arg0: vector<2xbf16>) -> vector<2xbf16> {
  // expected-error@+1 {{FTZ is required for vector<2xbf16>}}
  %0 = nvvm.ex2 %arg0 : vector<2xbf16>
  llvm.return %0 : vector<2xbf16>
}
