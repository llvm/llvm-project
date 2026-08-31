// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=triple=amdgpu9.08-amd-amdhsa --split-input-file --verify-diagnostics
// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=triple=amdgpu9.0a-amd-amdhsa --split-input-file --verify-diagnostics

// gfx908 and gfx90a have FeatureMAIInsts but no fp8 conversions at all, so the
// fp8 MFMAs -- which first appear on gfx942 -- must not be selected for them.

func.func @mfma_bf8(%arg0 : vector<8xf8E5M2FNUZ>, %arg1 : vector<4xf32>) {
  // expected-error@below {{op no intrinsic matching MFMA size on given chipset}}
  // expected-error@below {{failed to legalize operation 'amdgpu.mfma'}}
  amdgpu.mfma 16x16x32 %arg0 * %arg0 + %arg1 : vector<8xf8E5M2FNUZ>, vector<8xf8E5M2FNUZ>, vector<4xf32>
  func.return
}

// -----

func.func @mfma_fp8(%arg0 : vector<8xf8E4M3FNUZ>, %arg1 : vector<4xf32>) {
  // expected-error@below {{op no intrinsic matching MFMA size on given chipset}}
  // expected-error@below {{failed to legalize operation 'amdgpu.mfma'}}
  amdgpu.mfma 16x16x32 %arg0 * %arg0 + %arg1 : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<4xf32>
  func.return
}
