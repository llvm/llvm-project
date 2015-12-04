; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=R600 %s
; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=SI %s

; R600: {{^}}amdgpu_trunc:
; R600: TRUNC {{\*? *}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: {{^}}amdgpu_trunc:
; SI: v_trunc_f32

define void @amdgpu_trunc(float addrspace(1)* %out, float %x) {
entry:
  %0 = call float @llvm.AMDGPU.trunc(float %x)
  store float %0, float addrspace(1)* %out
  ret void
}

declare float @llvm.AMDGPU.trunc(float ) readnone
