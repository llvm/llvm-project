; RUN: llc -march=amdgcn < %s | FileCheck -check-prefix=SI --check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck -check-prefix=SI --check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG --check-prefix=FUNC %s

declare float @llvm.trunc.f32(float) nounwind readnone
declare <2 x float> @llvm.trunc.v2f32(<2 x float>) nounwind readnone
declare <3 x float> @llvm.trunc.v3f32(<3 x float>) nounwind readnone
declare <4 x float> @llvm.trunc.v4f32(<4 x float>) nounwind readnone
declare <8 x float> @llvm.trunc.v8f32(<8 x float>) nounwind readnone
declare <16 x float> @llvm.trunc.v16f32(<16 x float>) nounwind readnone

; FUNC-LABEL: {{^}}ftrunc_f32:
; EG: TRUNC
; SI: v_trunc_f32_e32
define amdgpu_kernel void @ftrunc_f32(ptr addrspace(1) %out, float %x) {
  %y = call float @llvm.trunc.f32(float %x) nounwind readnone
  store float %y, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}ftrunc_v2f32:
; EG: TRUNC
; EG: TRUNC
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
define amdgpu_kernel void @ftrunc_v2f32(ptr addrspace(1) %out, <2 x float> %x) {
  %y = call <2 x float> @llvm.trunc.v2f32(<2 x float> %x) nounwind readnone
  store <2 x float> %y, ptr addrspace(1) %out
  ret void
}

; FIXME-FUNC-LABEL: {{^}}ftrunc_v3f32:
; FIXME-EG: TRUNC
; FIXME-EG: TRUNC
; FIXME-EG: TRUNC
; FIXME-SI: v_trunc_f32_e32
; FIXME-SI: v_trunc_f32_e32
; FIXME-SI: v_trunc_f32_e32
; define amdgpu_kernel void @ftrunc_v3f32(ptr addrspace(1) %out, <3 x float> %x) {
;   %y = call <3 x float> @llvm.trunc.v3f32(<3 x float> %x) nounwind readnone
;   store <3 x float> %y, ptr addrspace(1) %out
;   ret void
; }

; FUNC-LABEL: {{^}}ftrunc_v4f32:
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
define amdgpu_kernel void @ftrunc_v4f32(ptr addrspace(1) %out, <4 x float> %x) {
  %y = call <4 x float> @llvm.trunc.v4f32(<4 x float> %x) nounwind readnone
  store <4 x float> %y, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}ftrunc_v8f32:
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
define amdgpu_kernel void @ftrunc_v8f32(ptr addrspace(1) %out, <8 x float> %x) {
  %y = call <8 x float> @llvm.trunc.v8f32(<8 x float> %x) nounwind readnone
  store <8 x float> %y, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}ftrunc_v16f32:
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; EG: TRUNC
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
; SI: v_trunc_f32_e32
define amdgpu_kernel void @ftrunc_v16f32(ptr addrspace(1) %out, <16 x float> %x) {
  %y = call <16 x float> @llvm.trunc.v16f32(<16 x float> %x) nounwind readnone
  store <16 x float> %y, ptr addrspace(1) %out
  ret void
}
