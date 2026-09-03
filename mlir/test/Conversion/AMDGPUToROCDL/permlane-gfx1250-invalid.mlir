// RUN: mlir-opt --convert-amdgpu-to-rocdl=triple=amdgpu12.50-amd-amdhsa --split-input-file --verify-diagnostics %s

// gfx1250 has FeaturePermlane16Swap but not FeaturePermlane32Swap; the 16-wide
// form is covered as a positive case in permlane.mlir.

func.func @permlane32(%arg0 : i32) -> i32 {
  // expected-error@below {{op permlane_swap of row length 32 is not supported}}
  // expected-error@below {{failed to legalize operation 'amdgpu.permlane_swap'}}
  %0 = amdgpu.permlane_swap %arg0 32 : i32
  return %0 : i32
}
