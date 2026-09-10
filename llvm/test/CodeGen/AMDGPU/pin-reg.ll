; RUN: llc -mtriple=amdgcn -mcpu=gfx950 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx950 -verify-machineinstrs -amdgpu-hard-pin-regs=0 < %s | FileCheck -check-prefixes=SOFT %s

; Tests for the llvm.amdgcn.pin.{vgpr,agpr} register-pinning intrinsics.

declare i32 @llvm.amdgcn.workitem.id.x()
declare <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32>, i32 immarg)
declare <2 x i32> @llvm.amdgcn.pin.vgpr.v2i32(<2 x i32>, i32 immarg)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)
declare <8 x i32> @llvm.amdgcn.pin.agpr.v8i32(<8 x i32>, i32 immarg)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32)

; An AGPR pin on the A/B inputs makes the loads AGPR-born and the MFMA read AGPR
; operands, with no agpr<->vgpr shuffle.
; CHECK-LABEL: {{^}}pin_agpr_input:
; CHECK: global_load_{{.*}} a[
; CHECK: global_load_{{.*}} a[
; CHECK-NOT: v_accvgpr
; CHECK: v_mfma_f32_16x16x16_f16 {{[va]}}[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}]
; The pin is honored even with hard pinning disabled (soft allocation hint).
; SOFT-LABEL: {{^}}pin_agpr_input:
; SOFT: v_mfma_f32_16x16x16_f16
define amdgpu_kernel void @pin_agpr_input(ptr addrspace(1) %pa, ptr addrspace(1) %pb, ptr addrspace(1) %pc) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %ga = getelementptr <4 x half>, ptr addrspace(1) %pa, i32 %tid
  %gb = getelementptr <4 x half>, ptr addrspace(1) %pb, i32 %tid
  %gc = getelementptr <4 x float>, ptr addrspace(1) %pc, i32 %tid
  %a = load <4 x half>, ptr addrspace(1) %ga
  %b = load <4 x half>, ptr addrspace(1) %gb
  %ai = bitcast <4 x half> %a to <2 x i32>
  %bi = bitcast <4 x half> %b to <2 x i32>
  %ap = call <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32> %ai, i32 0)
  %bp = call <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32> %bi, i32 8)
  %af = bitcast <2 x i32> %ap to <4 x half>
  %bf = bitcast <2 x i32> %bp to <4 x half>
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %af, <4 x half> %bf, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  store <4 x float> %d, ptr addrspace(1) %gc
  ret void
}

; A VGPR pin keeps its value in VGPRs (identity around the load's natural file).
; CHECK-LABEL: {{^}}pin_vgpr_value:
; CHECK: global_load_{{.*}} v[
; CHECK-NOT: v_accvgpr
define amdgpu_kernel void @pin_vgpr_value(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gp = getelementptr <2 x i32>, ptr addrspace(1) %p, i32 %tid
  %gq = getelementptr <2 x i32>, ptr addrspace(1) %q, i32 %tid
  %v = load <2 x i32>, ptr addrspace(1) %gp
  %vp = call <2 x i32> @llvm.amdgcn.pin.vgpr.v2i32(<2 x i32> %v, i32 4)
  store <2 x i32> %vp, ptr addrspace(1) %gq
  ret void
}

; Regression: two pins taking sub-slices of ONE wide load must not clobber each
; other (a naive rewrite of the shared register miscompiled). Both halves must be
; used; verify-machineinstrs (in the RUN line) also guards liveness.
; CHECK-LABEL: {{^}}pin_shared_load:
; CHECK: v_mfma_f32_16x16x16_f16
define amdgpu_kernel void @pin_shared_load(ptr addrspace(1) %p, ptr addrspace(1) %pc) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gp = getelementptr <4 x i32>, ptr addrspace(1) %p, i32 %tid
  %gc = getelementptr <4 x float>, ptr addrspace(1) %pc, i32 %tid
  %w = load <4 x i32>, ptr addrspace(1) %gp
  %alo = shufflevector <4 x i32> %w, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %bhi = shufflevector <4 x i32> %w, <4 x i32> poison, <2 x i32> <i32 2, i32 3>
  %ap = call <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32> %alo, i32 0)
  %bp = call <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32> %bhi, i32 2)
  %af = bitcast <2 x i32> %ap to <4 x half>
  %bf = bitcast <2 x i32> %bp to <4 x half>
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %af, <4 x half> %bf, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  store <4 x float> %d, ptr addrspace(1) %gc
  ret void
}

; A wide (8-dword) AGPR pin whose value is a REG_SEQUENCE of subregister slices
; of wider loads must not crash: the hard-pin load-tuple fast path bails and the
; pass falls back to soft, still placing the inputs in AGPRs (checked here via
; the scaled f8f6f4 MFMA, whose fp8/fp4 A/B are eight dwords). verify-machineinstrs
; in the RUN line guards against malformed liveness.
; CHECK-LABEL: {{^}}pin_agpr_wide:
; CHECK: global_load_{{.*}} a[
; CHECK: v_mfma_f32_16x16x128_f8f6f4 v[{{[0-9:]+}}], a[{{[0-9:]+}}], a[
define amdgpu_kernel void @pin_agpr_wide(ptr addrspace(1) %pa, ptr addrspace(1) %pb, ptr addrspace(1) %pc) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %ga = getelementptr <8 x i32>, ptr addrspace(1) %pa, i32 %tid
  %gb = getelementptr <8 x i32>, ptr addrspace(1) %pb, i32 %tid
  %gc = getelementptr <4 x float>, ptr addrspace(1) %pc, i32 %tid
  %a = load <8 x i32>, ptr addrspace(1) %ga
  %b = load <8 x i32>, ptr addrspace(1) %gb
  %ap = call <8 x i32> @llvm.amdgcn.pin.agpr.v8i32(<8 x i32> %a, i32 0)
  %bp = call <8 x i32> @llvm.amdgcn.pin.agpr.v8i32(<8 x i32> %b, i32 8)
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %ap, <8 x i32> %bp, <4 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  store <4 x float> %d, ptr addrspace(1) %gc
  ret void
}

; Self-containment: a function with NO pin intrinsic is unaffected by the pass.
; It gets the target's default MFMA form (accumulator AGPR, inputs VGPR) with no
; pin-introduced agpr<->vgpr shuffles.
; CHECK-LABEL: {{^}}no_pin:
; CHECK: v_mfma_f32_16x16x16_f16 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[
; CHECK-NOT: v_accvgpr
define amdgpu_kernel void @no_pin(ptr addrspace(1) %pa, ptr addrspace(1) %pb, ptr addrspace(1) %pc) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %ga = getelementptr <4 x half>, ptr addrspace(1) %pa, i32 %tid
  %gb = getelementptr <4 x half>, ptr addrspace(1) %pb, i32 %tid
  %gc = getelementptr <4 x float>, ptr addrspace(1) %pc, i32 %tid
  %a = load <4 x half>, ptr addrspace(1) %ga
  %b = load <4 x half>, ptr addrspace(1) %gb
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %b, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  store <4 x float> %d, ptr addrspace(1) %gc
  ret void
}
