// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx942 --split-input-file --verify-diagnostics

func.func @packed_trunc_ocp_type_requires_ocp_chipset(%arg0: f32) {
  // expected-error@below {{'amdgpu.packed_trunc_2xfp8' op no truncation to result type available on given chipset}}
  // expected-error@below {{failed to legalize operation 'amdgpu.packed_trunc_2xfp8'}}
  %0 = amdgpu.packed_trunc_2xfp8 %arg0, undef into undef[word 0] : f32 to vector<4xf8E4M3FN>
  return
}

// -----

func.func @packed_stoch_round_ocp_type_requires_ocp_chipset(%arg0: f32,
                                                            %arg1: i32) {
  // expected-error@below {{'amdgpu.packed_stoch_round_fp8' op no stochastic rounding to result type available on given chipset}}
  // expected-error@below {{failed to legalize operation 'amdgpu.packed_stoch_round_fp8'}}
  %0 = amdgpu.packed_stoch_round_fp8 %arg0 + %arg1 into undef[0] : f32 to vector<4xf8E4M3FN>
  return
}
