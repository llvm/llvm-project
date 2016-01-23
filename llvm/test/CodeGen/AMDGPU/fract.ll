; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=FUNC %s
; XUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare float @llvm.fabs.f32(float) #0
declare float @llvm.floor.f32(float) #0

; FUNC-LABEL: {{^}}fract_f32:
; CI: v_fract_f32_e32 [[RESULT:v[0-9]+]], [[INPUT:v[0-9]+]]
; SI: v_floor_f32_e32 [[FLR:v[0-9]+]], [[INPUT:v[0-9]+]]
; SI: v_subrev_f32_e32 [[RESULT:v[0-9]+]], [[FLR]], [[INPUT]]
; GCN: buffer_store_dword [[RESULT]]

; XEG: FRACT
define void @fract_f32(float addrspace(1)* %out, float addrspace(1)* %src) #1 {
  %x = load float, float addrspace(1)* %src
  %floor.x = call float @llvm.floor.f32(float %x)
  %fract = fsub float %x, %floor.x
  store float %fract, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fract_f32_neg:
; CI: v_fract_f32_e64 [[RESULT:v[0-9]+]], -[[INPUT:v[0-9]+]]
; SI: v_floor_f32_e64 [[FLR:v[0-9]+]], -[[INPUT:v[0-9]+]]
; SI: v_sub_f32_e64 [[RESULT:v[0-9]+]], -[[INPUT]], [[FLR]]
; GCN: buffer_store_dword [[RESULT]]

; XEG: FRACT
define void @fract_f32_neg(float addrspace(1)* %out, float addrspace(1)* %src) #1 {
  %x = load float, float addrspace(1)* %src
  %x.neg = fsub float -0.0, %x
  %floor.x.neg = call float @llvm.floor.f32(float %x.neg)
  %fract = fsub float %x.neg, %floor.x.neg
  store float %fract, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fract_f32_neg_abs:
; CI: v_fract_f32_e64 [[RESULT:v[0-9]+]], -|[[INPUT:v[0-9]+]]|
; SI: v_floor_f32_e64 [[FLR:v[0-9]+]], -|[[INPUT:v[0-9]+]]|
; SI: v_sub_f32_e64 [[RESULT:v[0-9]+]], -|[[INPUT]]|, [[FLR]]
; GCN: buffer_store_dword [[RESULT]]

; XEG: FRACT
define void @fract_f32_neg_abs(float addrspace(1)* %out, float addrspace(1)* %src) #1 {
  %x = load float, float addrspace(1)* %src
  %abs.x = call float @llvm.fabs.f32(float %x)
  %neg.abs.x = fsub float -0.0, %abs.x
  %floor.neg.abs.x = call float @llvm.floor.f32(float %neg.abs.x)
  %fract = fsub float %neg.abs.x, %floor.neg.abs.x
  store float %fract, float addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
