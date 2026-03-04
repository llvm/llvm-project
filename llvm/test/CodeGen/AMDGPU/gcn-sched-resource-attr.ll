; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -enable-post-misched=0 \
; RUN:     < %s | FileCheck -check-prefix=GCN %s

; Smoke test: the "amdgpu-sched-strategy"="resource" function attribute
; selects the gcn-resource pre-RA scheduling strategy
; (GCNPreRACriticalResource) without crashing on a function that contains
; reserved-resource (MFMA) consumers.

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>, i32, i32, i32)

define amdgpu_kernel void @resource_strategy(<4 x half> %a, <4 x half> %b,
                                             <4 x float> %c,
                                             ptr addrspace(1) %out) #0 {
; GCN-LABEL: resource_strategy:
; GCN: v_mfma_f32_16x16x16
  %r = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %b, <4 x float> %c, i32 0, i32 0, i32 0)
  store <4 x float> %r, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-sched-strategy"="resource" }
