// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// -----

func.func @fptrunc_narrower(%in : vector<16xf16>) {
  // expected-error @+1 {{'nvgpu.truncf' op result type 'f32' must be narrower than operand type 'f16'}}
  %out = nvgpu.truncf %in : vector<16xf16> to vector<16xf32>
  return
}

// -----

func.func @fptrunc_src_bitwidth(%in : vector<16xf8E5M2>) {
  // expected-error @+1 {{'nvgpu.truncf' op input type must be 64/32/16 bitwidth, but got 8}}
  %out = nvgpu.truncf %in : vector<16xf8E5M2> to vector<16xf4E2M1FN>
  return
}

// -----

func.func @fptrunc_e8m0_bad_rounding(%in : vector<16xf32>) {
  // expected-error @+1 {{'nvgpu.truncf' op expects RZ or RP rounding mode when result type is e8m0, but got #nvvm.fp_rnd_mode<rn>}}
  %out = nvgpu.truncf %in {rnd = #nvvm.fp_rnd_mode<rn>}
      : vector<16xf32> to vector<16xf8E8M0FNU>
  return
}

// -----

func.func @fptrunc_unsupported_sat_mode(%in : vector<8xf32>) {
  // expected-error @+1 {{'nvgpu.truncf' op attribute 'sat' failed to satisfy constraint: Describes the saturation mode whose value is one of {none, satfinite}}}
  %out = nvgpu.truncf %in {sat = #nvvm.sat_mode<sat>}
      : vector<8xf32> to vector<8xf8E4M3FN>
  return
}

// -----

func.func @fptrunc_f32_to_f8_rz(%in : vector<16xf32>) {
  // expected-error @+1 {{'nvgpu.truncf' op expects RN rounding mode, but got #nvvm.fp_rnd_mode<rz>}}
  %out = nvgpu.truncf %in {rnd = #nvvm.fp_rnd_mode<rz>}
      : vector<16xf32> to vector<16xf8E4M3FN>
  return
}

// -----

func.func @fptrunc_f64_to_f16_rz(%in : vector<4xf64>) {
  // expected-error @+1 {{'nvgpu.truncf' op expects RN rounding mode for f64 input, but got #nvvm.fp_rnd_mode<rz>}}
  %out = nvgpu.truncf %in {rnd = #nvvm.fp_rnd_mode<rz>}
      : vector<4xf64> to vector<4xf16>
  return
}

// -----

func.func @fptrunc_rs_unsupported_types(%in : vector<16xf32>) {
  // expected-error @+1 {{'nvgpu.truncf' op RS (stochastic) rounding is only supported for f32->f16/bf16, got 'f32' -> 'f8E4M3FN'}}
  %out = nvgpu.truncf %in {rnd = #nvvm.fp_rnd_mode<rs>}
      : vector<16xf32> to vector<16xf8E4M3FN>
  return
}

// -----

func.func @fptrunc_rs_no_random_bits(%in : vector<4xf32>) {
  // expected-error @+1 {{'nvgpu.truncf' op random_bits operand is required with RS rounding}}
  %out = nvgpu.truncf %in {rnd = #nvvm.fp_rnd_mode<rs>}
      : vector<4xf32> to vector<4xf16>
  return
}

// -----

func.func @fptrunc_bad_rounding(%in : vector<16xf32>) {
  // expected-error @+1 {{'nvgpu.truncf' op expects RN rounding mode, but got #nvvm.fp_rnd_mode<rp>}}
  %out = nvgpu.truncf %in {rnd = #nvvm.fp_rnd_mode<rp>}
      : vector<16xf32> to vector<16xf8E4M3FN>
  return
}

// -----

func.func @fptrunc_random_bits_no_rs(%in : vector<4xf32>, %rbits : i32) {
  // expected-error @+1 {{'nvgpu.truncf' op random_bits can only be used with RS rounding mode}}
  %out = nvgpu.truncf %in, %rbits
      : vector<4xf32> to vector<4xf16>
  return
}

// -----

func.func @fptrunc_shape_mismatch(%in : vector<16xf32>) {
  // expected-error @+1 {{'nvgpu.truncf' op input and output shapes must match}}
  %out = nvgpu.truncf %in : vector<16xf32> to vector<8xf8E4M3FN>
  return
}

// -----

func.func @fptrunc_scalar_vector_mismatch(%in : f32) {
  // expected-error @+1 {{'nvgpu.truncf' op input and output must both be scalars or both be vectors}}
  %out = nvgpu.truncf %in : f32 to vector<1xf16>
  return
}

// -----

func.func @fptrunc_rank0_vector(%in : vector<f32>) {
  // expected-error @+1 {{'nvgpu.truncf' op operand #0 must be floating-point or vector of floating-point values, but got 'vector<f32>'}}
  %out = nvgpu.truncf %in : vector<f32> to vector<f16>
  return
}
