; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK %s

; gfx1250 has 1024 addressable VGPRs. A pin to a VGPR index >= 256 is honored --
; the value is placed in the requested high tuple, reachable via the
; S_SET_VGPR_MSB addressing mode -- provided the tuple is inside the kernel's
; VGPR budget. A pin is a hint, and the allocation order stops at the budget the
; kernel's occupancy allows, so a high pin needs the kernel to declare that
; budget (amdgpu-num-vgpr plus a work-group size and waves-per-EU that leave
; room for it, i.e. __launch_bounds__ + __attribute__((amdgpu_num_vgpr))).
; Without the declaration the tuple is not in the order and the value is placed
; normally. That is attributes #0 below, on the kernels that pin above 256.

declare <4 x float> @llvm.amdgcn.pin.vgpr.v4f32(<4 x float>, i32 immarg)
declare <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32>, i32 immarg)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16(i1, <16 x bfloat>, i1, <16 x bfloat>, i16, <8 x float>, i1, i1)
declare i32 @llvm.amdgcn.workitem.id.x()

; CHECK-LABEL: {{^}}pin_high_vgpr:
; CHECK: s_set_vgpr_msb
; CHECK: v[{{[0-9:]+}}] /*v[300:303]*/
; CHECK: .set .Lpin_high_vgpr.num_vgpr, 304
define amdgpu_kernel void @pin_high_vgpr(ptr addrspace(1) %p) #0 {
  %v = load <4 x float>, ptr addrspace(1) %p
  %pv = call <4 x float> @llvm.amdgcn.pin.vgpr.v4f32(<4 x float> %v, i32 300)
  store <4 x float> %pv, ptr addrspace(1) %p
  ret void
}

; A WMMA B operand that two loads assemble is defined by a REG_SEQUENCE.
; Coalescing folds that into the hinted value, since the tuple is used as a
; whole, so the operand lands in the requested tuple.
; CHECK-LABEL: {{^}}pin_two_load_tuple:
; CHECK: v_wmma_f32_16x16x32_bf16 v[{{[0-9:]+}}], v[128:135],
define amdgpu_kernel void @pin_two_load_tuple(ptr addrspace(1) %o, ptr addrspace(1) %pa, ptr addrspace(1) %pb) #0 {
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
; lanes reach the REG_SEQUENCE as subregister slices of the wider load. Each
; load defines half the pinned value, and neither half is the pinned value, so
; the hint on the whole reaches neither def and the pair is placed normally. A
; value assembled this way has to be pinned at the width each load defines to be
; placed. The pin costs nothing here beyond going unmet -- note the VGPR count
; does not grow to cover the declared tuple.
; CHECK-LABEL: {{^}}pin_split_wide_load:
; CHECK: global_load_b128 v[0:3],
; CHECK: global_load_b128 v[4:7],
; CHECK: .set .Lpin_split_wide_load.num_vgpr, 9
define amdgpu_kernel void @pin_split_wide_load(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 {
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
; pins on the same tuple. Both are hints, and as in pin_split_wide_load the
; value is loaded in two halves, so neither is met and the variable is placed
; normally -- it does stay in place across the update, which is what the two
; pins were asking for.
; CHECK-LABEL: {{^}}pin_reassigned_variable:
; CHECK: v_pk_fma_f32 v[{{[0-9:]+}}], {{.*}}v[{{[0-9:]+}}], -1.0
; CHECK: .set .Lpin_reassigned_variable.num_vgpr, 9
define amdgpu_kernel void @pin_reassigned_variable(ptr addrspace(1) %p) #0 {
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

; Two distinct values whose live ranges overlap both ask for one tuple: the
; second one is still loaded when the first is read, so they cannot share it.
; One gets the tuple and the other is placed normally.
; CHECK-LABEL: {{^}}pin_two_live_values:
; CHECK: global_load_b128 v[2:5],
; CHECK: global_load_b128 v[{{[0-9:]+}}] /*v[300:303]*/
; CHECK: global_store_b128 v{{[0-9]+}}, v[2:5],
; CHECK: global_store_b128 v{{[0-9]+}}, v[{{[0-9:]+}}] /*v[300:303]*/
define amdgpu_kernel void @pin_two_live_values(ptr addrspace(1) %p, ptr addrspace(1) %q) #0 {
  %a = load volatile <4 x float>, ptr addrspace(1) %p
  %pa = call <4 x float> @llvm.amdgcn.pin.vgpr.v4f32(<4 x float> %a, i32 300)
  %b = load volatile <4 x float>, ptr addrspace(1) %q
  %pb = call <4 x float> @llvm.amdgcn.pin.vgpr.v4f32(<4 x float> %b, i32 300)
  store volatile <4 x float> %pa, ptr addrspace(1) %p
  store volatile <4 x float> %pb, ptr addrspace(1) %q
  ret void
}

; A pinned value defined outside a loop and carried into it reaches a PHI. The
; hint survives the copies that PHI elimination and coalescing introduce, so the
; accumulator lands on its tuple and accumulates in place -- and the VGPR count
; has to cover it. This is the shape a pinned matrix-multiply accumulator has,
; and the one a hint places most reliably: one live range, no split.
; CHECK-LABEL: {{^}}pin_into_loop_phi:
; CHECK: v_wmma_f32_16x16x32_bf16 v[108:115], v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[108:115]
; CHECK: s_endpgm
; CHECK: .set .Lpin_into_loop_phi.num_vgpr, 116
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

attributes #0 = { "amdgpu-flat-work-group-size"="128,128" "amdgpu-num-vgpr"="1024" "amdgpu-waves-per-eu"="1,1" }
