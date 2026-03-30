; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
; RUN:     -enable-post-misched=1 < %s | FileCheck -check-prefix=GCN %s

; Smoke test: the "amdgpu-post-sched-strategy"="resource" function attribute
; selects the GCNPostRACriticalResource post-RA scheduling strategy. The
; strategy attaches the resource distance map at post-RA scheduling time
; and must not crash on a function with reserved-resource (MFMA) consumers
; on multiple paths to the same accumulator.

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>, i32, i32, i32)

define amdgpu_kernel void @post_resource_strategy(<4 x half> %a, <4 x half> %b,
                                                  <4 x half> %a2, <4 x half> %b2,
                                                  <4 x float> %c,
                                                  ptr addrspace(1) %out) #0 {
; GCN-LABEL: post_resource_strategy:
; GCN: v_mfma_f32_16x16x16
; GCN: v_mfma_f32_16x16x16
  %r1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %b, <4 x float> %c, i32 0, i32 0, i32 0)
  %r2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a2, <4 x half> %b2, <4 x float> %r1, i32 0, i32 0, i32 0)
  store <4 x float> %r2, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-post-sched-strategy"="resource" }
