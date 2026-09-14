; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK %s

; gfx1250 has 1024 addressable VGPRs. A pin to a VGPR index >= 256 must be
; honored: the value is placed in the requested high tuple (reachable via the
; S_SET_VGPR_MSB addressing mode) and the VGPR count / occupancy cap grows to
; cover the pinned range.

declare <4 x float> @llvm.amdgcn.pin.vgpr.v4f32(<4 x float>, i32 immarg)
declare <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32>, i32 immarg)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16(i1, <16 x bfloat>, i1, <16 x bfloat>, i16, <8 x float>, i1, i1)
declare i32 @llvm.amdgcn.workitem.id.x()

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

; A 32-byte load off a divergent address is selected as two dwordx4 loads whose
; lanes reach the REG_SEQUENCE as subregister slices of the wider load. Placing
; a lane on its own would strand the rest of its load, so each load is placed as
; a whole onto its half of the pinned tuple. The stores read the pinned tuple
; directly: the REG_SEQUENCEs that reassemble each half are folded away rather
; than materialised as a copy per lane.
; CHECK-LABEL: {{^}}pin_split_wide_load:
; CHECK: global_load_b128 v[{{[0-9:]+}}] /*v[304:307]*/
; CHECK: global_load_b128 v[{{[0-9:]+}}] /*v[300:303]*/
; CHECK-NOT: v_mov
; CHECK: global_store_b128 v{{[0-9]+}}, v[{{[0-9:]+}}] /*v[304:307]*/
; CHECK: global_store_b128 v{{[0-9]+}}, v[{{[0-9:]+}}] /*v[300:303]*/
; CHECK: .set .Lpin_split_wide_load.num_vgpr, 308
define amdgpu_kernel void @pin_split_wide_load(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %idx = sext i32 %tid to i64
  %a = getelementptr inbounds <8 x i32>, ptr addrspace(1) %in, i64 %idx
  %v = load <8 x i32>, ptr addrspace(1) %a, align 32
  %pv = call <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32> %v, i32 300)
  %o = getelementptr inbounds <8 x i32>, ptr addrspace(1) %out, i64 %idx
  store <8 x i32> %pv, ptr addrspace(1) %o, align 32
  ret void
}

; Clang pins every store to a pinned variable, so reassigning one yields two
; pins on the same tuple. The second takes the tuple over -- the update is
; in-place, reading and writing the same registers -- instead of leaving the
; variable's later value wherever the allocator puts it.
; CHECK-LABEL: {{^}}pin_reassigned_variable:
; CHECK: global_load_b128 v[{{[0-9:]+}}] /*v[304:307]*/
; CHECK: global_load_b128 v[{{[0-9:]+}}] /*v[300:303]*/
; CHECK: v_pk_fma_f32 v[{{[0-9:]+}}] /*v[306:307]*/, {{.*}}v[{{[0-9:]+}}] /*v[306:307]*/
; CHECK: v_pk_fma_f32 v[{{[0-9:]+}}] /*v[300:301]*/, {{.*}}v[{{[0-9:]+}}] /*v[300:301]*/
; CHECK: .set .Lpin_reassigned_variable.num_vgpr, 308
define amdgpu_kernel void @pin_reassigned_variable(ptr addrspace(1) %p) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %idx = sext i32 %tid to i64
  %a = getelementptr inbounds <8 x float>, ptr addrspace(1) %p, i64 %idx
  %v = load <8 x float>, ptr addrspace(1) %a, align 32
  %v.i = bitcast <8 x float> %v to <8 x i32>
  %p1 = call <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32> %v.i, i32 300)
  %f = bitcast <8 x i32> %p1 to <8 x float>
  %m = fmul contract <8 x float> %f, splat (float 3.000000e+00)
  %s = fadd contract <8 x float> %m, splat (float -1.000000e+00)
  %s.i = bitcast <8 x float> %s to <8 x i32>
  %p2 = call <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32> %s.i, i32 300)
  store <8 x i32> %p2, ptr addrspace(1) %a, align 32
  ret void
}

; Two distinct values whose live ranges overlap must not share the tuple, even
; though both ask for it: the second one is still loaded when the first is read.
; CHECK-LABEL: {{^}}pin_two_live_values:
; CHECK: global_load_b128 v[{{[0-9:]+}}] /*v[300:303]*/
; CHECK: global_load_b128 v[0:3],
; CHECK: global_store_b128 v{{[0-9]+}}, v[{{[0-9:]+}}] /*v[300:303]*/
; CHECK: global_store_b128 v{{[0-9]+}}, v[0:3],
define amdgpu_kernel void @pin_two_live_values(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %a = load volatile <4 x float>, ptr addrspace(1) %p
  %pa = call <4 x float> @llvm.amdgcn.pin.vgpr.v4f32(<4 x float> %a, i32 300)
  %b = load volatile <4 x float>, ptr addrspace(1) %q
  %pb = call <4 x float> @llvm.amdgcn.pin.vgpr.v4f32(<4 x float> %b, i32 300)
  store volatile <4 x float> %pa, ptr addrspace(1) %p
  store volatile <4 x float> %pb, ptr addrspace(1) %q
  ret void
}

; A pinned value defined outside a loop and carried into it reaches a PHI. The
; physical tuple must not be substituted into the PHI operand: LiveVariables
; walks PHI sources through getVarInfo(), which asserts on a physical register,
; so the pin falls back to the soft path instead. Two pinned accumulators keep
; both PHIs live across the back edge. A soft hint costs no occupancy: the VGPR
; count reflects what the allocator used, not the range the pins asked for.
; CHECK-LABEL: {{^}}pin_into_loop_phi:
; CHECK: v_wmma_f32_16x16x32_bf16
; CHECK: s_endpgm
; CHECK: .set .Lpin_into_loop_phi.num_vgpr, 16
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
