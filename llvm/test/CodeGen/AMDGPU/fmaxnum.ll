; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_fmax_f32_ieee_mode_on:
; GCN: v_mul_f32_e64 [[QUIET0:v[0-9]+]], 1.0, s{{[0-9]+}}
; GCN: v_mul_f32_e64 [[QUIET1:v[0-9]+]], 1.0, s{{[0-9]+}}
; GCN: v_max_f32_e32 [[RESULT:v[0-9]+]], [[QUIET1]], [[QUIET0]]
; GCN-NOT: [[RESULT]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_fmax_f32_ieee_mode_on(ptr addrspace(1) %out, float %a, float %b) #0 {
  %val = call float @llvm.maxnum.f32(float %a, float %b) #1
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fmax_f32_ieee_mode_off:
; GCN: v_max_f32_e32 v0, v0, v1
; GCN-NEXT: ; return
define amdgpu_ps float @test_fmax_f32_ieee_mode_off(float %a, float %b) #0 {
  %val = call float @llvm.maxnum.f32(float %a, float %b) #1
  ret float %val
}

; GCN-LABEL: {{^}}test_fmax_v2f32:
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
define amdgpu_kernel void @test_fmax_v2f32(ptr addrspace(1) %out, <2 x float> %a, <2 x float> %b) #0 {
  %val = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %a, <2 x float> %b)
  store <2 x float> %val, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}test_fmax_v3f32:
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN-NOT: v_max_f32
define amdgpu_kernel void @test_fmax_v3f32(ptr addrspace(1) %out, <3 x float> %a, <3 x float> %b) nounwind {
  %val = call <3 x float> @llvm.maxnum.v3f32(<3 x float> %a, <3 x float> %b) #0
  store <3 x float> %val, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}test_fmax_v4f32:
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
define amdgpu_kernel void @test_fmax_v4f32(ptr addrspace(1) %out, <4 x float> %a, <4 x float> %b) #0 {
  %val = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %a, <4 x float> %b)
  store <4 x float> %val, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}test_fmax_v8f32:
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
define amdgpu_kernel void @test_fmax_v8f32(ptr addrspace(1) %out, <8 x float> %a, <8 x float> %b) #0 {
  %val = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %a, <8 x float> %b)
  store <8 x float> %val, ptr addrspace(1) %out, align 32
  ret void
}

; GCN-LABEL: {{^}}test_fmax_v16f32:
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
define amdgpu_kernel void @test_fmax_v16f32(ptr addrspace(1) %out, <16 x float> %a, <16 x float> %b) #0 {
  %val = call <16 x float> @llvm.maxnum.v16f32(<16 x float> %a, <16 x float> %b)
  store <16 x float> %val, ptr addrspace(1) %out, align 64
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmax_f32:
; GCN-NOT: v_max_f32_e32
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 2.0
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmax_f32(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.maxnum.f32(float 1.0, float 2.0)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmax_f32_nan_nan:
; GCN-NOT: v_max_f32_e32
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7fc00000
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmax_f32_nan_nan(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.maxnum.f32(float 0x7FF8000000000000, float 0x7FF8000000000000)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmax_f32_val_nan:
; GCN-NOT: v_max_f32_e32
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 1.0
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmax_f32_val_nan(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.maxnum.f32(float 1.0, float 0x7FF8000000000000)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmax_f32_nan_val:
; GCN-NOT: v_max_f32_e32
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 1.0
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmax_f32_nan_val(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.maxnum.f32(float 0x7FF8000000000000, float 1.0)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmax_f32_p0_p0:
; GCN-NOT: v_max_f32_e32
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmax_f32_p0_p0(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.maxnum.f32(float 0.0, float 0.0)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmax_f32_p0_n0:
; GCN-NOT: v_max_f32_e32
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmax_f32_p0_n0(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.maxnum.f32(float 0.0, float -0.0)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmax_f32_n0_p0:
; GCN-NOT: v_max_f32_e32
; GCN: v_bfrev_b32_e32 [[REG:v[0-9]+]], 1{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmax_f32_n0_p0(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.maxnum.f32(float -0.0, float 0.0)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmax_f32_n0_n0:
; GCN-NOT: v_max_f32_e32
; GCN: v_bfrev_b32_e32 [[REG:v[0-9]+]], 1{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmax_f32_n0_n0(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.maxnum.f32(float -0.0, float -0.0)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fmax_var_immediate_f32_no_ieee:
; GCN: v_max_f32_e64 {{v[0-9]+}}, {{s[0-9]+}}, 2.0
define amdgpu_ps float @fmax_var_immediate_f32_no_ieee(float inreg %a) #0 {
  %val = call float @llvm.maxnum.f32(float %a, float 2.0) #0
  ret float %val
}

; GCN-LABEL: {{^}}fmax_immediate_var_f32_no_ieee:
; GCN: v_max_f32_e64 {{v[0-9]+}}, {{s[0-9]+}}, 2.0
define amdgpu_ps float @fmax_immediate_var_f32_no_ieee(float inreg %a) #0 {
  %val = call float @llvm.maxnum.f32(float 2.0, float %a) #0
  ret float %val
}

; GCN-LABEL: {{^}}fmax_var_literal_f32_no_ieee:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x42c60000
; GCN: v_max_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, [[REG]]
define amdgpu_ps float @fmax_var_literal_f32_no_ieee(float inreg %a) #0 {
  %val = call float @llvm.maxnum.f32(float %a, float 99.0) #0
  ret float %val
}

; GCN-LABEL: {{^}}fmax_literal_var_f32_no_ieee:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x42c60000
; GCN: v_max_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, [[REG]]
define amdgpu_ps float @fmax_literal_var_f32_no_ieee(float inreg %a) #0 {
  %val = call float @llvm.maxnum.f32(float 99.0, float %a) #0
  ret float %val
}

; GCN-LABEL: {{^}}test_func_fmax_v3f32:
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN: v_max_f32_e32
; GCN-NOT: v_max_f32
define <3 x float> @test_func_fmax_v3f32(<3 x float> %a, <3 x float> %b) #0 {
  %val = call <3 x float> @llvm.maxnum.v3f32(<3 x float> %a, <3 x float> %b) #0
  ret <3 x float> %val
}

declare float @llvm.maxnum.f32(float, float) #1
declare <2 x float> @llvm.maxnum.v2f32(<2 x float>, <2 x float>) #1
declare <3 x float> @llvm.maxnum.v3f32(<3 x float>, <3 x float>) #1
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>) #1
declare <8 x float> @llvm.maxnum.v8f32(<8 x float>, <8 x float>) #1
declare <16 x float> @llvm.maxnum.v16f32(<16 x float>, <16 x float>) #1
declare double @llvm.maxnum.f64(double, double)

attributes #0 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
attributes #1 = { nounwind readnone }
