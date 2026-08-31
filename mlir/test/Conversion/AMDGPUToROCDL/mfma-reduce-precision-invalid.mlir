// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=triple=amdgpu9.50-amd-amdhsa --split-input-file --verify-diagnostics

// The xf32 MFMAs come from FeatureXF32Insts, which only gfx942 has. gfx950
// compares greater than gfx942 by ISA version, so a version-ordered check let
// them through here.

func.func @mfma_reduce_precision_32x32x4(%arg0 : vector<2xf32>,
                                         %arg1 : vector<16xf32>) {
  // expected-error@below {{op no intrinsic matching MFMA size on given chipset}}
  // expected-error@below {{failed to legalize operation 'amdgpu.mfma'}}
  amdgpu.mfma 32x32x4 %arg0 * %arg0 + %arg1 reducePrecision : vector<2xf32>, vector<2xf32>, vector<16xf32>
  func.return
}

// -----

func.func @mfma_reduce_precision_16x16x8(%arg0 : vector<2xf32>,
                                         %arg1 : vector<4xf32>) {
  // expected-error@below {{op no intrinsic matching MFMA size on given chipset}}
  // expected-error@below {{failed to legalize operation 'amdgpu.mfma'}}
  amdgpu.mfma 16x16x8 %arg0 * %arg0 + %arg1 reducePrecision : vector<2xf32>, vector<2xf32>, vector<4xf32>
  func.return
}
