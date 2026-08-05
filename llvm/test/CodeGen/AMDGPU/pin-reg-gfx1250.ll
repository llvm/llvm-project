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
