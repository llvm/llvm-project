; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare float @llvm.fabs.f32(float  %Val)
declare float @llvm.AMDGPU.fract.f32(float) nounwind readnone

; FUNC-LABEL: {{^}}fract_f32:
; CI: v_fract_f32_e32 [[RESULT:v[0-9]+]], [[INPUT:v[0-9]+]]
; SI: v_floor_f32_e32 [[FLR:v[0-9]+]], [[INPUT:v[0-9]+]]
; SI: v_subrev_f32_e32 [[RESULT:v[0-9]+]], [[FLR]], [[INPUT]]
; GCN: buffer_store_dword [[RESULT]]
; EG: FRACT
define void @fract_f32(float addrspace(1)* %out, float addrspace(1)* %src) nounwind {
  %val = load float, float addrspace(1)* %src, align 4
  %fract = call float @llvm.AMDGPU.fract.f32(float %val) nounwind readnone
  store float %fract, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}fract_f32_neg:
; CI: v_fract_f32_e64 [[RESULT:v[0-9]+]], -[[INPUT:v[0-9]+]]
; SI: v_floor_f32_e64 [[FLR:v[0-9]+]], -[[INPUT:v[0-9]+]]
; SI: v_sub_f32_e64 [[RESULT:v[0-9]+]], -[[INPUT]], [[FLR]]
; GCN: buffer_store_dword [[RESULT]]
; EG: FRACT
define void @fract_f32_neg(float addrspace(1)* %out, float addrspace(1)* %src) nounwind {
  %val = load float, float addrspace(1)* %src, align 4
  %neg = fsub float 0.0, %val
  %fract = call float @llvm.AMDGPU.fract.f32(float %neg) nounwind readnone
  store float %fract, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}fract_f32_neg_abs:
; CI: v_fract_f32_e64 [[RESULT:v[0-9]+]], -|[[INPUT:v[0-9]+]]|
; SI: v_floor_f32_e64 [[FLR:v[0-9]+]], -|[[INPUT:v[0-9]+]]|
; SI: v_sub_f32_e64 [[RESULT:v[0-9]+]], -|[[INPUT]]|, [[FLR]]
; GCN: buffer_store_dword [[RESULT]]
; EG: FRACT
define void @fract_f32_neg_abs(float addrspace(1)* %out, float addrspace(1)* %src) nounwind {
  %val = load float, float addrspace(1)* %src, align 4
  %abs = call float @llvm.fabs.f32(float %val)
  %neg = fsub float 0.0, %abs
  %fract = call float @llvm.AMDGPU.fract.f32(float %neg) nounwind readnone
  store float %fract, float addrspace(1)* %out, align 4
  ret void
}
