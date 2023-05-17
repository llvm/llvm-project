; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck %s -check-prefix=R600 -check-prefix=FUNC

; FUNC-LABEL: {{^}}rint_f32:
; R600: RNDNE

; SI: v_rndne_f32_e32
define amdgpu_kernel void @rint_f32(ptr addrspace(1) %out, float %in) {
entry:
  %0 = call float @llvm.rint.f32(float %in) #0
  store float %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}rint_v2f32:
; R600: RNDNE
; R600: RNDNE

; SI: v_rndne_f32_e32
; SI: v_rndne_f32_e32
define amdgpu_kernel void @rint_v2f32(ptr addrspace(1) %out, <2 x float> %in) {
entry:
  %0 = call <2 x float> @llvm.rint.v2f32(<2 x float> %in) #0
  store <2 x float> %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}rint_v4f32:
; R600: RNDNE
; R600: RNDNE
; R600: RNDNE
; R600: RNDNE

; SI: v_rndne_f32_e32
; SI: v_rndne_f32_e32
; SI: v_rndne_f32_e32
; SI: v_rndne_f32_e32
define amdgpu_kernel void @rint_v4f32(ptr addrspace(1) %out, <4 x float> %in) {
entry:
  %0 = call <4 x float> @llvm.rint.v4f32(<4 x float> %in) #0
  store <4 x float> %0, ptr addrspace(1) %out
  ret void
}

declare float @llvm.rint.f32(float) #0
declare <2 x float> @llvm.rint.v2f32(<2 x float>) #0
declare <4 x float> @llvm.rint.v4f32(<4 x float>) #0

attributes #0 = { nounwind readnone }
