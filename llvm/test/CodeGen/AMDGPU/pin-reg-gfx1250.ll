; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK %s

; gfx1250 has 1024 addressable VGPRs. A pin to a VGPR index >= 256 must be
; honored: the value is placed in the requested high tuple (reachable via the
; S_SET_VGPR_MSB addressing mode) and the VGPR count / occupancy cap grows to
; cover the pinned range.

declare <4 x float> @llvm.amdgcn.pin.vgpr.v4f32(<4 x float>, i32 immarg)
declare <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32>, i32 immarg)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16(i1, <16 x bfloat>, i1, <16 x bfloat>, i16, <8 x float>, i1, i1)

; CHECK-LABEL: {{^}}pin_high_vgpr:
; CHECK: s_set_vgpr_msb
; CHECK: v[{{[0-9:]+}}] /*v[300:303]*/
; CHECK: .set .Lpin_high_vgpr.num_vgpr, 304
define amdgpu_kernel void @pin_high_vgpr(ptr addrspace(1) %p) {
  %v = load <4 x float>, ptr addrspace(1) %p
  %pv = call <4 x float> @llvm.amdgcn.pin.vgpr.v4f32(<4 x float> %v, i32 300)
  store <4 x float> %pv, ptr addrspace(1) %p
  ret void
}

; A WMMA B operand that two loads assemble is defined by a REG_SEQUENCE, which
; the general path rejects outright. The load-tuple path now takes VGPR pins as
; well, so the operand lands in the requested tuple instead of wherever the
; allocator puts it. Only the AGPR form of this path ran before, and gfx1250 has
; no AGPR file, so such a pin was always dropped here.
; CHECK-LABEL: {{^}}pin_two_load_tuple:
; CHECK: v_wmma_f32_16x16x32_bf16 v[{{[0-9:]+}}], v[128:135],
define amdgpu_kernel void @pin_two_load_tuple(ptr addrspace(1) %o, ptr addrspace(1) %pa, ptr addrspace(1) %pb) {
  %p1 = getelementptr <8 x bfloat>, ptr addrspace(1) %pb, i64 1
  %b0 = load <8 x bfloat>, ptr addrspace(1) %pb, align 16
  %b1 = load <8 x bfloat>, ptr addrspace(1) %p1, align 16
  %b = shufflevector <8 x bfloat> %b0, <8 x bfloat> %b1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %b.i = bitcast <16 x bfloat> %b to <8 x i32>
  %p = call <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32> %b.i, i32 128)
  %pb.v = bitcast <8 x i32> %p to <16 x bfloat>
  %a = load <16 x bfloat>, ptr addrspace(1) %pa, align 32
  %d = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16(i1 false, <16 x bfloat> %pb.v, i1 false, <16 x bfloat> %a, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  store <8 x float> %d, ptr addrspace(1) %o
  ret void
}

; A pinned value defined outside a loop and carried into it reaches a PHI. The
; physical tuple must not be substituted into the PHI operand: LiveVariables
; walks PHI sources through getVarInfo(), which asserts on a physical register,
; so the pin falls back to the soft path instead. Two pinned accumulators keep
; both PHIs live across the back edge.
; CHECK-LABEL: {{^}}pin_into_loop_phi:
; CHECK: v_wmma_f32_16x16x32_bf16
; CHECK: s_endpgm
define amdgpu_kernel void @pin_into_loop_phi(ptr addrspace(1) %o) {
entry:
  %i0 = call <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32> zeroinitializer, i32 100)
  %i1 = call <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32> zeroinitializer, i32 108)
  br label %loop

loop:
  %a1 = phi <8 x i32> [ %i1, %entry ], [ %d.i, %loop ]
  %a0 = phi <8 x i32> [ %i0, %entry ], [ zeroinitializer, %loop ]
  %c = bitcast <8 x i32> %a1 to <8 x float>
  %d = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16(i1 false, <16 x bfloat> zeroinitializer, i1 false, <16 x bfloat> zeroinitializer, i16 0, <8 x float> %c, i1 false, i1 false)
  %d.i = bitcast <8 x float> %d to <8 x i32>
  br i1 false, label %exit, label %loop

exit:
  %o1 = getelementptr <8 x i32>, ptr addrspace(1) %o, i32 1
  store <8 x i32> %a0, ptr addrspace(1) %o
  store <8 x i32> %a1, ptr addrspace(1) %o1
  ret void
}
