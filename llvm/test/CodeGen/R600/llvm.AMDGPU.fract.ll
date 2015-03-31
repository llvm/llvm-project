; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare float @llvm.AMDGPU.fract.f32(float) nounwind readnone

; Legacy name
declare float @llvm.AMDIL.fraction.f32(float) nounwind readnone

; FUNC-LABEL: {{^}}fract_f32:
; SI: v_fract_f32
; EG: FRACT
define void @fract_f32(float addrspace(1)* %out, float addrspace(1)* %src) nounwind {
  %val = load float addrspace(1)* %src, align 4
  %fract = call float @llvm.AMDGPU.fract.f32(float %val) nounwind readnone
  store float %fract, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}fract_f32_legacy_amdil:
; SI: v_fract_f32
; EG: FRACT
define void @fract_f32_legacy_amdil(float addrspace(1)* %out, float addrspace(1)* %src) nounwind {
  %val = load float addrspace(1)* %src, align 4
  %fract = call float @llvm.AMDIL.fraction.f32(float %val) nounwind readnone
  store float %fract, float addrspace(1)* %out, align 4
  ret void
}
