// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// -----

func.func @fpext_wider(%in : vector<16xf8E5M2>) {
  // expected-error @+1 {{'nvgpu.convert.fpext' op result type 'f4E2M1FN' must be wider than operand type 'f8E5M2'}}
  %out = nvgpu.convert.fpext %in : vector<16xf8E5M2> to vector<16xf4E2M1FN>
  return
}

// -----

func.func @fpext_dst_bitwidth(%in : vector<16xf4E2M1FN>) {
  // expected-error @+1 {{'nvgpu.convert.fpext' op result type must be 16, 32, or 64 bitwidth, but got 8}}
  %out = nvgpu.convert.fpext %in : vector<16xf4E2M1FN> to vector<16xf8E4M3FN>
  return
}

// -----

func.func @fpext_e8m0_to_f16(%in : vector<16xf8E8M0FNU>) {
  // expected-error @+1 {{'nvgpu.convert.fpext' op expects bf16 or f32 output type when input type is e8m0.}}
  %out = nvgpu.convert.fpext %in : vector<16xf8E8M0FNU> to vector<16xf16>
  return
}

// -----

func.func @fpext_bad_rounding(%in : vector<16xf8E5M2>) {
  // expected-error @+1 {{'nvgpu.convert.fpext' op expects RN rounding mode, but got #nvvm.fp_rnd_mode<rz>}}
  %out = nvgpu.convert.fpext %in {rnd = #nvvm.fp_rnd_mode<rz>}
      : vector<16xf8E5M2> to vector<16xf16>
  return
}

// -----

func.func @fpext_relu_bf16(%in : vector<8xf8E5M2>) {
  // expected-error @+1 {{'nvgpu.convert.fpext' op relu is not supported for bf16 destination}}
  %out = nvgpu.convert.fpext %in {relu = true} : vector<8xf8E5M2> to vector<8xbf16>
  return
}

// -----

func.func @fpext_shape_mismatch(%in : vector<16xf8E5M2>) {
  // expected-error @+1 {{'nvgpu.convert.fpext' op input and output shapes must match}}
  %out = nvgpu.convert.fpext %in : vector<16xf8E5M2> to vector<8xf16>
  return
}

// -----

func.func @fpext_scalar_vector_mismatch(%in : f8E4M3FN) {
  // expected-error @+1 {{'nvgpu.convert.fpext' op input and output must both be scalars or both be vectors/tensors}}
  %out = nvgpu.convert.fpext %in : f8E4M3FN to vector<1xf16>
  return
}

// -----

func.func @fpext_rank0_tensor(%in : tensor<f8E4M3FN>) {
  // expected-error @+1 {{'nvgpu.convert.fpext' op rank-0 shaped types are not supported, use scalar type instead}}
  %out = nvgpu.convert.fpext %in : tensor<f8E4M3FN> to tensor<f16>
  return
}

// -----

func.func @fpext_container_mismatch(%in : vector<4xf8E4M3FN>) {
  // expected-error @+1 {{'nvgpu.convert.fpext' op input and output must be the same container type (both vector or both tensor)}}
  %out = nvgpu.convert.fpext %in : vector<4xf8E4M3FN> to tensor<4xf16>
  return
}

// -----

func.func @fpext_unranked_tensor(%in : tensor<*xf8E4M3FN>) {
  // expected-error @+1 {{'nvgpu.convert.fpext' op unranked tensor types are not supported}}
  %out = nvgpu.convert.fpext %in : tensor<*xf8E4M3FN> to tensor<*xf16>
  return
}
