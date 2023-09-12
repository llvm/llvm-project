; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=SI -check-prefix=FUNC %s

declare float @llvm.amdgcn.rcp.f32(float) #0
declare double @llvm.amdgcn.rcp.f64(double) #0

declare double @llvm.amdgcn.sqrt.f64(double) #0
declare float @llvm.amdgcn.sqrt.f32(float) #0
declare double @llvm.sqrt.f64(double) #0
declare float @llvm.sqrt.f32(float) #0

; FUNC-LABEL: {{^}}rcp_undef_f32:
; SI: v_mov_b32_e32 [[NAN:v[0-9]+]], 0x7fc00000
; SI-NOT: [[NAN]]
; SI: buffer_store_dword [[NAN]]
define amdgpu_kernel void @rcp_undef_f32(ptr addrspace(1) %out) #1 {
  %rcp = call float @llvm.amdgcn.rcp.f32(float undef)
  store float %rcp, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_2_f32:
; SI-NOT: v_rcp_f32
; SI: v_mov_b32_e32 v{{[0-9]+}}, 0.5
define amdgpu_kernel void @rcp_2_f32(ptr addrspace(1) %out) #1 {
  %rcp = call float @llvm.amdgcn.rcp.f32(float 2.0)
  store float %rcp, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_10_f32:
; SI-NOT: v_rcp_f32
; SI: v_mov_b32_e32 v{{[0-9]+}}, 0x3dcccccd
define amdgpu_kernel void @rcp_10_f32(ptr addrspace(1) %out) #1 {
  %rcp = call float @llvm.amdgcn.rcp.f32(float 10.0)
  store float %rcp, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}safe_no_fp32_denormals_rcp_f32:
; SI: v_rcp_f32_e32 [[RESULT:v[0-9]+]], s{{[0-9]+}}
; SI-NOT: [[RESULT]]
; SI: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @safe_no_fp32_denormals_rcp_f32(ptr addrspace(1) %out, float %src) #1 {
  %rcp = fdiv float 1.0, %src, !fpmath !0
  store float %rcp, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}safe_f32_denormals_rcp_pat_f32:
; SI: v_rcp_f32_e32 [[RESULT:v[0-9]+]], s{{[0-9]+}}
; SI-NOT: [[RESULT]]
; SI: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @safe_f32_denormals_rcp_pat_f32(ptr addrspace(1) %out, float %src) #4 {
  %rcp = fdiv float 1.0, %src, !fpmath !0
  store float %rcp, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}unsafe_f32_denormals_rcp_pat_f32:
; SI: v_div_scale_f32
define amdgpu_kernel void @unsafe_f32_denormals_rcp_pat_f32(ptr addrspace(1) %out, float %src) #3 {
  %rcp = fdiv float 1.0, %src
  store float %rcp, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}safe_rsq_rcp_pat_f32:
; SI: v_mul_f32
; SI: v_rsq_f32
; SI: v_mul_f32
; SI: v_fma_f32
; SI: v_fma_f32
; SI: v_fma_f32
; SI: v_fma_f32
; SI: v_fma_f32
; SI: v_rcp_f32
define amdgpu_kernel void @safe_rsq_rcp_pat_f32(ptr addrspace(1) %out, float %src) #1 {
  %sqrt = call contract float @llvm.sqrt.f32(float %src)
  %rcp = call contract float @llvm.amdgcn.rcp.f32(float %sqrt)
  store float %rcp, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}safe_rsq_rcp_pat_amdgcn_sqrt_f32:
; SI: v_sqrt_f32_e32
; SI: v_rcp_f32_e32
define amdgpu_kernel void @safe_rsq_rcp_pat_amdgcn_sqrt_f32(ptr addrspace(1) %out, float %src) #1 {
  %sqrt = call contract float @llvm.amdgcn.sqrt.f32(float %src)
  %rcp = call contract float @llvm.amdgcn.rcp.f32(float %sqrt)
  store float %rcp, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}safe_rsq_rcp_pat_amdgcn_sqrt_f32_nocontract:
; SI: v_sqrt_f32_e32
; SI: v_rcp_f32_e32
define amdgpu_kernel void @safe_rsq_rcp_pat_amdgcn_sqrt_f32_nocontract(ptr addrspace(1) %out, float %src) #1 {
  %sqrt = call float @llvm.amdgcn.sqrt.f32(float %src)
  %rcp = call contract float @llvm.amdgcn.rcp.f32(float %sqrt)
  store float %rcp, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}unsafe_rsq_rcp_pat_f32:
; SI: v_sqrt_f32_e32
; SI: v_rcp_f32_e32
define amdgpu_kernel void @unsafe_rsq_rcp_pat_f32(ptr addrspace(1) %out, float %src) #2 {
  %sqrt = call float @llvm.sqrt.f32(float %src)
  %rcp = call float @llvm.amdgcn.rcp.f32(float %sqrt)
  store float %rcp, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_f64:
; SI: v_rcp_f64_e32 [[RESULT:v\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}
; SI-NOT: [[RESULT]]
; SI: buffer_store_dwordx2 [[RESULT]]
define amdgpu_kernel void @rcp_f64(ptr addrspace(1) %out, double %src) #1 {
  %rcp = call double @llvm.amdgcn.rcp.f64(double %src)
  store double %rcp, ptr addrspace(1) %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}unsafe_rcp_f64:
; SI: v_rcp_f64_e32 [[RESULT:v\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}
; SI-NOT: [[RESULT]]
; SI: buffer_store_dwordx2 [[RESULT]]
define amdgpu_kernel void @unsafe_rcp_f64(ptr addrspace(1) %out, double %src) #2 {
  %rcp = call double @llvm.amdgcn.rcp.f64(double %src)
  store double %rcp, ptr addrspace(1) %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}rcp_pat_f64:
; SI: v_div_scale_f64
define amdgpu_kernel void @rcp_pat_f64(ptr addrspace(1) %out, double %src) #1 {
  %rcp = fdiv double 1.0, %src
  store double %rcp, ptr addrspace(1) %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}unsafe_rcp_pat_f64:
; SI: v_rcp_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_fma_f64
define amdgpu_kernel void @unsafe_rcp_pat_f64(ptr addrspace(1) %out, double %src) #2 {
  %rcp = fdiv double 1.0, %src
  store double %rcp, ptr addrspace(1) %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}safe_rsq_rcp_pat_f64:
; SI-NOT: v_rsq_f64_e32
; SI: v_rsq_f64
; SI: v_mul_f64
; SI: v_mul_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_rcp_f64
define amdgpu_kernel void @safe_rsq_rcp_pat_f64(ptr addrspace(1) %out, double %src) #1 {
  %sqrt = call double @llvm.sqrt.f64(double %src)
  %rcp = call double @llvm.amdgcn.rcp.f64(double %sqrt)
  store double %rcp, ptr addrspace(1) %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}safe_amdgcn_sqrt_rsq_rcp_pat_f64:
; SI-NOT: v_rsq_f64_e32
; SI: v_sqrt_f64
; SI: v_rcp_f64
define amdgpu_kernel void @safe_amdgcn_sqrt_rsq_rcp_pat_f64(double addrspace(1)* %out, double %src) #1 {
  %sqrt = call double @llvm.amdgcn.sqrt.f64(double %src)
  %rcp = call double @llvm.amdgcn.rcp.f64(double %sqrt)
  store double %rcp, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}unsafe_rsq_rcp_pat_f64:
; SI: v_rsq_f64
; SI: v_mul_f64
; SI: v_mul_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_rcp_f64
; SI: buffer_store_dwordx2
define amdgpu_kernel void @unsafe_rsq_rcp_pat_f64(double addrspace(1)* %out, double %src) #2 {
  %sqrt = call double @llvm.sqrt.f64(double %src)
  %rcp = call double @llvm.amdgcn.rcp.f64(double %sqrt)
  store double %rcp, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}unsafe_amdgcn_sqrt_rsq_rcp_pat_f64:
; SI: v_sqrt_f64_e32 [[SQRT:v\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}
; SI: v_rcp_f64_e32 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[SQRT]]
; SI: buffer_store_dwordx2 [[RESULT]]
define amdgpu_kernel void @unsafe_amdgcn_sqrt_rsq_rcp_pat_f64(ptr addrspace(1) %out, double %src) #2 {
  %sqrt = call double @llvm.amdgcn.sqrt.f64(double %src)
  %rcp = call double @llvm.amdgcn.rcp.f64(double %sqrt)
  store double %rcp, ptr addrspace(1) %out, align 8
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "unsafe-fp-math"="false" "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
attributes #2 = { nounwind "unsafe-fp-math"="true" "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
attributes #3 = { nounwind "unsafe-fp-math"="false" "denormal-fp-math-f32"="ieee,ieee" }
attributes #4 = { nounwind "unsafe-fp-math"="true" "denormal-fp-math-f32"="ieee,ieee" }

!0 = !{float 2.500000e+00}
