; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 < %s | FileCheck %s

; v32f16 FMINIMUMNUM/FMAXIMUMNUM is marked Custom, so in non-IEEE mode it
; must be split by lowerFMINIMUMNUM_FMAXIMUMNUM instead of falling through
; to selection, which previously failed with "Cannot select".

; CHECK-LABEL: min_v32f16_no_ieee:
; CHECK-COUNT-16: v_pk_min_f16
define amdgpu_kernel void @min_v32f16_no_ieee(ptr addrspace(1) %p, <32 x half> %a, <32 x half> %b) #0 {
  %r = call <32 x half> @llvm.minimumnum.v32f16(<32 x half> %a, <32 x half> %b)
  store <32 x half> %r, ptr addrspace(1) %p, align 64
  ret void
}

; CHECK-LABEL: max_v32f16_no_ieee:
; CHECK-COUNT-16: v_pk_max_f16
define amdgpu_kernel void @max_v32f16_no_ieee(ptr addrspace(1) %p, <32 x half> %a, <32 x half> %b) #0 {
  %r = call <32 x half> @llvm.maximumnum.v32f16(<32 x half> %a, <32 x half> %b)
  store <32 x half> %r, ptr addrspace(1) %p, align 64
  ret void
}

attributes #0 = { "amdgpu-ieee"="false" }
