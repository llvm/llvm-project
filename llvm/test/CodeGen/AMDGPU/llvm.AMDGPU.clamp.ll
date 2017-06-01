; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare float @llvm.fabs.f32(float) nounwind readnone
declare float @llvm.AMDGPU.clamp.f32(float, float, float) nounwind readnone

; FUNC-LABEL: {{^}}clamp_0_1_f32:
; SI: s_load_dword [[ARG:s[0-9]+]],
; SI: v_add_f32_e64 [[RESULT:v[0-9]+]], [[ARG]], 0 clamp{{$}}
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm

; EG: MOV_SAT
define void @clamp_0_1_f32(float addrspace(1)* %out, float %src) nounwind {
  %clamp = call float @llvm.AMDGPU.clamp.f32(float %src, float 0.0, float 1.0) nounwind readnone
  store float %clamp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}clamp_fabs_0_1_f32:
; SI: s_load_dword [[ARG:s[0-9]+]],
; SI: v_add_f32_e64 [[RESULT:v[0-9]+]], |[[ARG]]|, 0 clamp{{$}}
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @clamp_fabs_0_1_f32(float addrspace(1)* %out, float %src) nounwind {
  %src.fabs = call float @llvm.fabs.f32(float %src) nounwind readnone
  %clamp = call float @llvm.AMDGPU.clamp.f32(float %src.fabs, float 0.0, float 1.0) nounwind readnone
  store float %clamp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}clamp_fneg_0_1_f32:
; SI: s_load_dword [[ARG:s[0-9]+]],
; SI: v_add_f32_e64 [[RESULT:v[0-9]+]], -[[ARG]], 0 clamp{{$}}
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @clamp_fneg_0_1_f32(float addrspace(1)* %out, float %src) nounwind {
  %src.fneg = fsub float -0.0, %src
  %clamp = call float @llvm.AMDGPU.clamp.f32(float %src.fneg, float 0.0, float 1.0) nounwind readnone
  store float %clamp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}clamp_fneg_fabs_0_1_f32:
; SI: s_load_dword [[ARG:s[0-9]+]],
; SI: v_add_f32_e64 [[RESULT:v[0-9]+]], -|[[ARG]]|, 0 clamp{{$}}
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @clamp_fneg_fabs_0_1_f32(float addrspace(1)* %out, float %src) nounwind {
  %src.fabs = call float @llvm.fabs.f32(float %src) nounwind readnone
  %src.fneg.fabs = fsub float -0.0, %src.fabs
  %clamp = call float @llvm.AMDGPU.clamp.f32(float %src.fneg.fabs, float 0.0, float 1.0) nounwind readnone
  store float %clamp, float addrspace(1)* %out, align 4
  ret void
}
