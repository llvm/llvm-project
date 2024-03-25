; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_fmin_f32_ieee_mode_on:
; GCN: v_mul_f32_e64 [[QUIET0:v[0-9]+]], 1.0, s{{[0-9]+}}
; GCN: v_mul_f32_e64 [[QUIET1:v[0-9]+]], 1.0, s{{[0-9]+}}
; GCN: v_min_f32_e32 [[RESULT:v[0-9]+]], [[QUIET1]], [[QUIET0]]
; GCN-NOT: [[RESULT]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_fmin_f32_ieee_mode_on(ptr addrspace(1) %out, float %a, float %b) #0 {
  %val = call float @llvm.minnum.f32(float %a, float %b) #1
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fmin_nnan_f32_ieee_mode_on:
; GCN: s_waitcnt
; GCN-NEXT: v_min_f32_e32 v0, v0, v1
; GCN-NEXT: s_setpc_b64
define float @test_fmin_nnan_f32_ieee_mode_on(float %a, float %b) #0 {
  %val = call nnan float @llvm.minnum.f32(float %a, float %b) #1
  ret float %val
}

; GCN-LABEL: {{^}}test_fmin_nnan_f32_ieee_mode_off:
; GCN-NOT: v0
; GCN-NOT: v1
; GCN: v_min_f32_e32 v0, v0, v1
; GCN-NEXT: ; return
define amdgpu_ps float @test_fmin_nnan_f32_ieee_mode_off(float %a, float %b) #0 {
  %val = call nnan float @llvm.minnum.f32(float %a, float %b) #1
  ret float %val
}

; GCN-LABEL: {{^}}test_fmin_f32_ieee_mode_off:
; GCN: v_min_f32_e32 v0, v0, v1
; GCN-NEXT: ; return
define amdgpu_ps float @test_fmin_f32_ieee_mode_off(float %a, float %b) #0 {
  %val = call float @llvm.minnum.f32(float %a, float %b) #1
  ret float %val
}

; GCN-LABEL: {{^}}test_fmin_v2f32:
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
define amdgpu_kernel void @test_fmin_v2f32(ptr addrspace(1) %out, <2 x float> %a, <2 x float> %b) #0 {
  %val = call <2 x float> @llvm.minnum.v2f32(<2 x float> %a, <2 x float> %b)
  store <2 x float> %val, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}test_fmin_v4f32:
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
define amdgpu_kernel void @test_fmin_v4f32(ptr addrspace(1) %out, <4 x float> %a, <4 x float> %b) #0 {
  %val = call <4 x float> @llvm.minnum.v4f32(<4 x float> %a, <4 x float> %b)
  store <4 x float> %val, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}test_fmin_v8f32:
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
define amdgpu_kernel void @test_fmin_v8f32(ptr addrspace(1) %out, <8 x float> %a, <8 x float> %b) #0 {
  %val = call <8 x float> @llvm.minnum.v8f32(<8 x float> %a, <8 x float> %b)
  store <8 x float> %val, ptr addrspace(1) %out, align 32
  ret void
}

; GCN-LABEL: {{^}}test_fmin_v16f32:
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
define amdgpu_kernel void @test_fmin_v16f32(ptr addrspace(1) %out, <16 x float> %a, <16 x float> %b) #0 {
  %val = call <16 x float> @llvm.minnum.v16f32(<16 x float> %a, <16 x float> %b)
  store <16 x float> %val, ptr addrspace(1) %out, align 64
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmin_f32:
; GCN-NOT: v_min_f32_e32
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 1.0
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmin_f32(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.minnum.f32(float 1.0, float 2.0)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmin_f32_nan_nan:
; GCN-NOT: v_min_f32_e32
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7fc00000
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmin_f32_nan_nan(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.minnum.f32(float 0x7FF8000000000000, float 0x7FF8000000000000)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmin_f32_val_nan:
; GCN-NOT: v_min_f32_e32
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 1.0
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmin_f32_val_nan(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.minnum.f32(float 1.0, float 0x7FF8000000000000)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmin_f32_nan_val:
; GCN-NOT: v_min_f32_e32
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 1.0
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmin_f32_nan_val(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.minnum.f32(float 0x7FF8000000000000, float 1.0)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmin_f32_p0_p0:
; GCN-NOT: v_min_f32_e32
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmin_f32_p0_p0(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.minnum.f32(float 0.0, float 0.0)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmin_f32_p0_n0:
; GCN-NOT: v_min_f32_e32
; GCN: v_bfrev_b32_e32 [[REG:v[0-9]+]], 1{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmin_f32_p0_n0(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.minnum.f32(float 0.0, float -0.0)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmin_f32_n0_p0:
; GCN-NOT: v_min_f32_e32
; GCN: v_bfrev_b32_e32 [[REG:v[0-9]+]], 1{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmin_f32_n0_p0(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.minnum.f32(float -0.0, float 0.0)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}constant_fold_fmin_f32_n0_n0:
; GCN-NOT: v_min_f32_e32
; GCN: v_bfrev_b32_e32 [[REG:v[0-9]+]], 1{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @constant_fold_fmin_f32_n0_n0(ptr addrspace(1) %out) #0 {
  %val = call float @llvm.minnum.f32(float -0.0, float -0.0)
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fmin_var_immediate_f32_no_ieee:
; GCN: v_min_f32_e32 v0, 2.0, v0
define amdgpu_ps float @fmin_var_immediate_f32_no_ieee(float %a) #0 {
  %val = call float @llvm.minnum.f32(float %a, float 2.0) #1
  ret float %val
}

; GCN-LABEL: {{^}}fmin_immediate_var_f32_no_ieee:
; GCN: v_min_f32_e64 {{v[0-9]+}}, {{s[0-9]+}}, 2.0
define amdgpu_ps float @fmin_immediate_var_f32_no_ieee(float inreg %a) #0 {
  %val = call float @llvm.minnum.f32(float 2.0, float %a) #1
  ret float %val
}

; GCN-LABEL: {{^}}fmin_var_literal_f32_no_ieee:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x42c60000
; GCN: v_min_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, [[REG]]
define amdgpu_ps float @fmin_var_literal_f32_no_ieee(float inreg %a) #0 {
  %val = call float @llvm.minnum.f32(float %a, float 99.0) #1
  ret float %val
}

; GCN-LABEL: {{^}}fmin_literal_var_f32_no_ieee:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x42c60000
; GCN: v_min_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, [[REG]]
define amdgpu_ps float @fmin_literal_var_f32_no_ieee(float inreg %a) #0 {
  %val = call float @llvm.minnum.f32(float 99.0, float %a) #1
  ret float %val
}

; GCN-LABEL: {{^}}test_func_fmin_v3f32:
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN: v_min_f32_e32
; GCN-NOT: v_min_f32
define <3 x float> @test_func_fmin_v3f32(<3 x float> %a, <3 x float> %b) nounwind {
  %val = call <3 x float> @llvm.minnum.v3f32(<3 x float> %a, <3 x float> %b) #0
  ret <3 x float> %val
}

declare float @llvm.minnum.f32(float, float) #1
declare <2 x float> @llvm.minnum.v2f32(<2 x float>, <2 x float>) #1
declare <3 x float> @llvm.minnum.v3f32(<3 x float>, <3 x float>) #1
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>) #1
declare <8 x float> @llvm.minnum.v8f32(<8 x float>, <8 x float>) #1
declare <16 x float> @llvm.minnum.v16f32(<16 x float>, <16 x float>) #1

attributes #0 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
attributes #1 = { nounwind readnone }
