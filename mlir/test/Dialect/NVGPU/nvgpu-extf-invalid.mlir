// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// -----

func.func @fpext_wider(%in : vector<16xf8E5M2>) {
  // expected-error @+1 {{'nvgpu.extf' op result type 'f4E2M1FN' must be wider than operand type 'f8E5M2'}}
  %out = nvgpu.extf %in : vector<16xf8E5M2> to vector<16xf4E2M1FN>
  return
}

// -----

func.func @fpext_dst_bitwidth(%in : vector<16xf4E2M1FN>) {
  // expected-error @+1 {{'nvgpu.extf' op result type must be 16, 32, or 64 bitwidth, but got 8}}
  %out = nvgpu.extf %in : vector<16xf4E2M1FN> to vector<16xf8E4M3FN>
  return
}

// -----

func.func @fpext_e8m0_to_f16(%in : vector<16xf8E8M0FNU>) {
  // expected-error @+1 {{'nvgpu.extf' op expects bf16 or f32 output type when input type is e8m0.}}
  %out = nvgpu.extf %in : vector<16xf8E8M0FNU> to vector<16xf16>
  return
}

// -----

func.func @fpext_bad_rounding(%in : vector<16xf8E5M2>) {
  // expected-error @+1 {{'nvgpu.extf' op expects RN rounding mode, but got #nvvm.fp_rnd_mode<rz>}}
  %out = nvgpu.extf %in {rnd = #nvvm.fp_rnd_mode<rz>}
      : vector<16xf8E5M2> to vector<16xf16>
  return
}

// -----

func.func @fpext_relu_bf16(%in : vector<8xf8E5M2>) {
  // expected-error @+1 {{'nvgpu.extf' op relu is not supported for bf16 destination}}
  %out = nvgpu.extf %in {relu = true} : vector<8xf8E5M2> to vector<8xbf16>
  return
}

// -----

func.func @fpext_shape_mismatch(%in : vector<16xf8E5M2>) {
  // expected-error @+1 {{'nvgpu.extf' op input and output shapes must match}}
  %out = nvgpu.extf %in : vector<16xf8E5M2> to vector<8xf16>
  return
}

// -----

func.func @fpext_scalar_vector_mismatch(%in : f8E4M3FN) {
  // expected-error @+1 {{'nvgpu.extf' op input and output must both be scalars or both be vectors}}
  %out = nvgpu.extf %in : f8E4M3FN to vector<1xf16>
  return
}

// -----

func.func @fpext_rank0_vector(%in : vector<f8E4M3FN>) {
  // expected-error @+1 {{'nvgpu.extf' op operand #0 must be floating-point or vector of floating-point values, but got 'vector<f8E4M3FN>'}}
  %out = nvgpu.extf %in : vector<f8E4M3FN> to vector<f16>
  return
}
