; RUN: llc -mtriple=amdgcn -mcpu=gfx950 -stop-after=finalize-isel < %s | FileCheck -check-prefix=ISEL %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx950 < %s | FileCheck -check-prefix=GCN %s

; An MFMA whose src2 or result is strictly pinned by sched.agpr is selected in
; the AGPR form even when the function is not known to need AGPRs, so the
; accumulator stays in place instead of being copied to and from VGPRs around
; the MFMA. Soft pins and unpinned MFMAs keep the default VGPR form.

declare <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float>, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>, i32, i32, i32)

; ISEL-LABEL: name: strict_pin
; ISEL: V_MFMA_F32_16X16X16F16_e64
; ISEL: V_MFMA_F32_16X16X16F16_e64
; ISEL-NOT: V_MFMA_F32_16X16X16F16_vgprcd_e64
; GCN-LABEL: {{^}}strict_pin:
; GCN-NOT: v_accvgpr_read
; GCN: v_mfma_f32_16x16x16_f16 a[8:11], v[{{[0-9:]+}}], v[{{[0-9:]+}}], a[8:11]
; GCN-NOT: v_accvgpr_read
; GCN: v_mfma_f32_16x16x16_f16 a[8:11], v[{{[0-9:]+}}], v[{{[0-9:]+}}], a[8:11]
define amdgpu_kernel void @strict_pin(ptr addrspace(1) %out, <4 x half> %a, <4 x half> %b, <4 x float> %c) #0 {
  %cin = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %c, i32 8, i32 1)
  %r = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %b, <4 x float> %cin, i32 0, i32 0, i32 0)
  %rp = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %r, i32 8, i32 1)
  %cin2 = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %rp, i32 8, i32 1)
  %r2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %b, <4 x half> %a, <4 x float> %cin2, i32 0, i32 0, i32 0)
  %rp2 = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %r2, i32 8, i32 1)
  store <4 x float> %rp2, ptr addrspace(1) %out
  ret void
}

; ISEL-LABEL: name: soft_pin
; ISEL: V_MFMA_F32_16X16X16F16_vgprcd_e64
define amdgpu_kernel void @soft_pin(ptr addrspace(1) %out, <4 x half> %a, <4 x half> %b, <4 x float> %c) #0 {
  %cin = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %c, i32 8, i32 0)
  %r = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %b, <4 x float> %cin, i32 0, i32 0, i32 0)
  store <4 x float> %r, ptr addrspace(1) %out
  ret void
}

; ISEL-LABEL: name: unpinned
; ISEL: V_MFMA_F32_16X16X16F16_vgprcd_e64
define amdgpu_kernel void @unpinned(ptr addrspace(1) %out, <4 x half> %a, <4 x half> %b, <4 x float> %c) #0 {
  %r = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %b, <4 x float> %c, i32 0, i32 0, i32 0)
  store <4 x float> %r, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" }
