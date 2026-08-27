// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx1170 --split-input-file --verify-diagnostics

// gfx11.7 has FeatureOCPFP8ConversionInsts, so these conversions are available
// on it. They are rejected today because the predicate deciding whether a
// target uses the OCP fp8 formats is written as the version range "gfx9.5+ or
// gfx12+", which skips over gfx11.7 entirely.

func.func @ext_packed_fp8(%v: vector<4xf8E4M3FN>) -> f32 {
  // expected-error@below {{failed to legalize operation 'amdgpu.ext_packed_fp8'}}
  %ret = amdgpu.ext_packed_fp8 %v[0] : vector<4xf8E4M3FN> to f32
  func.return %ret : f32
}

// -----

func.func @ext_packed_bf8(%v: vector<4xf8E5M2>) -> f32 {
  // expected-error@below {{failed to legalize operation 'amdgpu.ext_packed_fp8'}}
  %ret = amdgpu.ext_packed_fp8 %v[0] : vector<4xf8E5M2> to f32
  func.return %ret : f32
}

// -----

func.func @packed_trunc_2xfp8(%v: f32) -> vector<4xf8E4M3FN> {
  // expected-error@below {{failed to legalize operation 'amdgpu.packed_trunc_2xfp8'}}
  %ret = amdgpu.packed_trunc_2xfp8 %v, undef into undef[word 0] : f32 to vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}
