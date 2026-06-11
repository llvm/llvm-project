// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// -----

func.func @convert_float_same_bitwidth(%in : vector<16xf16>) {
  // expected-error @+1 {{'nvgpu.convert.float' op input and output types must have different bitwidths, got 'f16' and 'bf16'}}
  %out = nvgpu.convert.float %in : vector<16xf16> to vector<16xbf16>
  return
}

// -----

func.func @fptrunc_src_bitwidth(%in : vector<16xf8E5M2>) {
  // expected-error @+1 {{'nvgpu.convert.float' op input type must be 64/32/16 bitwidth, but got 8}}
  %out = nvgpu.convert.float %in : vector<16xf8E5M2> to vector<16xf4E2M1FN>
  return
}

// -----

func.func @fptrunc_e8m0_bad_rounding(%in : vector<16xf32>) {
  // expected-error @+1 {{'nvgpu.convert.float' op expects RZ or RP rounding mode when result type is e8m0, but got #nvvm.fp_rnd_mode<rn>}}
  %out = nvgpu.convert.float %in {rnd = #nvvm.fp_rnd_mode<rn>}
      : vector<16xf32> to vector<16xf8E8M0FNU>
  return
}

// -----

func.func @fptrunc_unsupported_sat_mode(%in : vector<8xf32>) {
  // expected-error @+1 {{'nvgpu.convert.float' op attribute 'sat' failed to satisfy constraint: Describes the saturation mode whose value is one of {none, satfinite}}}
  %out = nvgpu.convert.float %in {sat = #nvvm.sat_mode<sat>}
      : vector<8xf32> to vector<8xf8E4M3FN>
  return
}

// -----

func.func @fptrunc_f32_to_f8_rz(%in : vector<16xf32>) {
  // expected-error @+1 {{'nvgpu.convert.float' op expects RN rounding mode, but got #nvvm.fp_rnd_mode<rz>}}
  %out = nvgpu.convert.float %in {rnd = #nvvm.fp_rnd_mode<rz>}
      : vector<16xf32> to vector<16xf8E4M3FN>
  return
}

// -----

func.func @fptrunc_f64_to_f16_rz(%in : vector<4xf64>) {
  // expected-error @+1 {{'nvgpu.convert.float' op expects RN rounding mode for f64 input, but got #nvvm.fp_rnd_mode<rz>}}
  %out = nvgpu.convert.float %in {rnd = #nvvm.fp_rnd_mode<rz>}
      : vector<4xf64> to vector<4xf16>
  return
}

// -----

func.func @fptrunc_rs_unsupported_types(%in : vector<16xf32>) {
  // expected-error @+1 {{'nvgpu.convert.float' op RS (stochastic) rounding is only supported for f32->f16/bf16, got 'f32' -> 'f8E4M3FN'}}
  %out = nvgpu.convert.float %in {rnd = #nvvm.fp_rnd_mode<rs>}
      : vector<16xf32> to vector<16xf8E4M3FN>
  return
}

// -----

func.func @fptrunc_rs_no_random_bits(%in : vector<4xf32>) {
  // expected-error @+1 {{'nvgpu.convert.float' op random_bits operand is required with RS rounding}}
  %out = nvgpu.convert.float %in {rnd = #nvvm.fp_rnd_mode<rs>}
      : vector<4xf32> to vector<4xf16>
  return
}

// -----

func.func @fptrunc_bad_rounding(%in : vector<16xf32>) {
  // expected-error @+1 {{'nvgpu.convert.float' op expects RN rounding mode, but got #nvvm.fp_rnd_mode<rp>}}
  %out = nvgpu.convert.float %in {rnd = #nvvm.fp_rnd_mode<rp>}
      : vector<16xf32> to vector<16xf8E4M3FN>
  return
}

// -----

func.func @fptrunc_random_bits_no_rs(%in : vector<4xf32>, %rbits : i32) {
  // expected-error @+1 {{'nvgpu.convert.float' op random_bits can only be used with RS rounding mode}}
  %out = nvgpu.convert.float %in, %rbits
      : vector<4xf32> to vector<4xf16>
  return
}

// -----

func.func @fptrunc_shape_mismatch(%in : vector<16xf32>) {
  // expected-error @+1 {{'nvgpu.convert.float' op input and output shapes must match}}
  %out = nvgpu.convert.float %in : vector<16xf32> to vector<8xf8E4M3FN>
  return
}

// -----

func.func @fptrunc_scalar_vector_mismatch(%in : f32) {
  // expected-error @+1 {{'nvgpu.convert.float' op input and output must both be scalars or both be vectors/tensors}}
  %out = nvgpu.convert.float %in : f32 to vector<1xf16>
  return
}

// -----

func.func @fptrunc_rank0_tensor(%in : tensor<f32>) {
  // expected-error @+1 {{'nvgpu.convert.float' op rank-0 shaped types are not supported, use scalar type instead}}
  %out = nvgpu.convert.float %in : tensor<f32> to tensor<f16>
  return
}

// -----

func.func @fptrunc_container_mismatch(%in : vector<4xf32>) {
  // expected-error @+1 {{'nvgpu.convert.float' op input and output must be the same container type (both vector or both tensor)}}
  %out = nvgpu.convert.float %in : vector<4xf32> to tensor<4xf16>
  return
}

// -----

func.func @fptrunc_unranked_tensor(%in : tensor<*xf32>) {
  // expected-error @+1 {{'nvgpu.convert.float' op unranked tensor types are not supported}}
  %out = nvgpu.convert.float %in : tensor<*xf32> to tensor<*xf16>
  return
}
