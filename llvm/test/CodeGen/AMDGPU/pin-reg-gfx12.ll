; RUN: llc -mtriple=amdgcn -mcpu=gfx1201 -verify-machineinstrs < %s | FileCheck %s

; RDNA4 (gfx1201) has only a VGPR file and uses WMMA. pin_vgpr must place the
; operands in the requested VGPRs; pin_agpr has no AGPR file to target and must
; degrade to a soft no-op (compile cleanly, no AGPRs, correct WMMA) rather than
; fail register allocation.

; pin_vgpr: A -> v[8:11], B -> v[12:15], D -> v[20:27], loaded straight in.
; CHECK-LABEL: pin_vgpr_wmma:
; CHECK: global_load_b128 v[8:11],
; CHECK: global_load_b128 v[12:15],
; CHECK: v_wmma_f32_16x16x16_f16 v[20:27], v[8:11], v[12:15]
define protected amdgpu_kernel void @pin_vgpr_wmma(ptr addrspace(1) nocapture readonly %A, ptr addrspace(1) nocapture readonly %B, ptr addrspace(1) nocapture writeonly %C) {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %off = zext i32 %id to i64
  %pa = getelementptr inbounds <8 x half>, ptr addrspace(1) %A, i64 %off
  %la = load <4 x i32>, ptr addrspace(1) %pa, align 16
  %pina = tail call <4 x i32> @llvm.amdgcn.pin.vgpr.v4i32(<4 x i32> %la, i32 8)
  %a = bitcast <4 x i32> %pina to <8 x half>
  %pb = getelementptr inbounds <8 x half>, ptr addrspace(1) %B, i64 %off
  %lb = load <4 x i32>, ptr addrspace(1) %pb, align 16
  %pinb = tail call <4 x i32> @llvm.amdgcn.pin.vgpr.v4i32(<4 x i32> %lb, i32 12)
  %b = bitcast <4 x i32> %pinb to <8 x half>
  %d = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %a, <8 x half> %b, <8 x float> zeroinitializer)
  %di = bitcast <8 x float> %d to <8 x i32>
  %pind = tail call <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32> %di, i32 20)
  %pc = getelementptr inbounds <8 x float>, ptr addrspace(1) %C, i64 %off
  store <8 x i32> %pind, ptr addrspace(1) %pc, align 32
  ret void
}

; pin_agpr on gfx1201: soft no-op. Compiles to a plain WMMA, no AGPRs used.
; CHECK-LABEL: pin_agpr_noop:
; CHECK-NOT: a[
; CHECK: v_wmma_f32_16x16x16_f16 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
; CHECK-NOT: a[
; CHECK: .set {{\.?L?}}pin_agpr_noop.num_agpr, 0
define protected amdgpu_kernel void @pin_agpr_noop(ptr addrspace(1) nocapture readonly %A, ptr addrspace(1) nocapture readonly %B, ptr addrspace(1) nocapture writeonly %C) {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %off = zext i32 %id to i64
  %pa = getelementptr inbounds <8 x half>, ptr addrspace(1) %A, i64 %off
  %la = load <4 x i32>, ptr addrspace(1) %pa, align 16
  %pina = tail call <4 x i32> @llvm.amdgcn.pin.agpr.v4i32(<4 x i32> %la, i32 0)
  %a = bitcast <4 x i32> %pina to <8 x half>
  %pb = getelementptr inbounds <8 x half>, ptr addrspace(1) %B, i64 %off
  %lb = load <4 x i32>, ptr addrspace(1) %pb, align 16
  %pinb = tail call <4 x i32> @llvm.amdgcn.pin.agpr.v4i32(<4 x i32> %lb, i32 4)
  %b = bitcast <4 x i32> %pinb to <8 x half>
  %d = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %a, <8 x half> %b, <8 x float> zeroinitializer)
  %pc = getelementptr inbounds <8 x float>, ptr addrspace(1) %C, i64 %off
  store <8 x float> %d, ptr addrspace(1) %pc, align 32
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half>, <8 x half>, <8 x float>)
declare <4 x i32> @llvm.amdgcn.pin.vgpr.v4i32(<4 x i32>, i32 immarg)
declare <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32>, i32 immarg)
declare <4 x i32> @llvm.amdgcn.pin.agpr.v4i32(<4 x i32>, i32 immarg)
