; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=FUNC %s

declare float @llvm.amdgcn.rsq.clamp.f32(float) #1
declare double @llvm.amdgcn.rsq.clamp.f64(double) #1

; FUNC-LABEL: {{^}}rsq_clamp_f32:
; SI: v_rsq_clamp_f32_e32

; VI: s_load_dword [[SRC:s[0-9]+]]
; VI-DAG: v_rsq_f32_e32 [[RSQ:v[0-9]+]], [[SRC]]
; VI-DAG: v_min_f32_e32 [[MIN:v[0-9]+]], 0x7f7fffff, [[RSQ]]
; VI: v_max_f32_e32 [[RESULT:v[0-9]+]], 0xff7fffff, [[MIN]]
; VI: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @rsq_clamp_f32(float addrspace(1)* %out, float %src) #0 {
  %rsq_clamp = call float @llvm.amdgcn.rsq.clamp.f32(float %src)
  store float %rsq_clamp, float addrspace(1)* %out
  ret void
}


; FUNC-LABEL: {{^}}rsq_clamp_f64:
; SI: v_rsq_clamp_f64_e32

; TODO: this constant should be folded:
; VI-DAG: s_mov_b32 [[NEG1:s[0-9]+]], -1
; VI-DAG: s_mov_b32 s[[LOW1:[0-9]+]], [[NEG1]]
; VI-DAG: s_mov_b32 s[[HIGH1:[0-9]+]], 0x7fefffff
; VI-DAG: s_mov_b32 s[[HIGH2:[0-9]+]], 0xffefffff
; VI-DAG: v_rsq_f64_e32 [[RSQ:v\[[0-9]+:[0-9]+\]]], s[{{[0-9]+:[0-9]+}}
; VI-DAG: v_min_f64 v[0:1], [[RSQ]], s{{\[}}[[LOW1]]:[[HIGH1]]]
; VI-DAG: v_max_f64 v[0:1], v[0:1], s{{\[}}[[LOW1]]:[[HIGH2]]]
define amdgpu_kernel void @rsq_clamp_f64(double addrspace(1)* %out, double %src) #0 {
  %rsq_clamp = call double @llvm.amdgcn.rsq.clamp.f64(double %src)
  store double %rsq_clamp, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}rsq_clamp_undef_f32:
; SI-NOT: v_rsq_clamp_f32
define amdgpu_kernel void @rsq_clamp_undef_f32(float addrspace(1)* %out) #0 {
  %rsq_clamp = call float @llvm.amdgcn.rsq.clamp.f32(float undef)
  store float %rsq_clamp, float addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
