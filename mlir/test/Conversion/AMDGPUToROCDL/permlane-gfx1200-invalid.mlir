// RUN: mlir-opt --convert-amdgpu-to-rocdl=triple=amdgpu12.00-amd-amdhsa --split-input-file --verify-diagnostics %s

// gfx1200 has neither FeaturePermlane16Swap nor FeaturePermlane32Swap, but
// compares greater than gfx950 by ISA version, so a version-ordered check
// accepted both widths.

func.func @permlane16(%arg0 : i32) -> i32 {
  // expected-error@below {{op permlane_swap of row length 16 is not supported}}
  // expected-error@below {{failed to legalize operation 'amdgpu.permlane_swap'}}
  %0 = amdgpu.permlane_swap %arg0 16 : i32
  return %0 : i32
}

// -----

func.func @permlane32(%arg0 : i32) -> i32 {
  // expected-error@below {{op permlane_swap of row length 32 is not supported}}
  // expected-error@below {{failed to legalize operation 'amdgpu.permlane_swap'}}
  %0 = amdgpu.permlane_swap %arg0 32 : i32
  return %0 : i32
}
