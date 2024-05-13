; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -mtriple=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}floor_f32:
; SI: v_floor_f32_e32
; R600: FLOOR
define amdgpu_kernel void @floor_f32(ptr addrspace(1) %out, float %in) {
  %tmp = call float @llvm.floor.f32(float %in) #0
  store float %tmp, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}floor_v2f32:
; SI: v_floor_f32_e32
; SI: v_floor_f32_e32

define amdgpu_kernel void @floor_v2f32(ptr addrspace(1) %out, <2 x float> %in) {
  %tmp = call <2 x float> @llvm.floor.v2f32(<2 x float> %in) #0
  store <2 x float> %tmp, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}floor_v4f32:
; SI: v_floor_f32_e32
; SI: v_floor_f32_e32
; SI: v_floor_f32_e32
; SI: v_floor_f32_e32

; R600: FLOOR
; R600: FLOOR
; R600: FLOOR
; R600: FLOOR
define amdgpu_kernel void @floor_v4f32(ptr addrspace(1) %out, <4 x float> %in) {
  %tmp = call <4 x float> @llvm.floor.v4f32(<4 x float> %in) #0
  store <4 x float> %tmp, ptr addrspace(1) %out
  ret void
}

; Function Attrs: nounwind readonly
declare float @llvm.floor.f32(float) #0

; Function Attrs: nounwind readonly
declare <2 x float> @llvm.floor.v2f32(<2 x float>) #0

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.floor.v4f32(<4 x float>) #0

attributes #0 = { nounwind readnone }
