; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_mfma_loop_zeroinit:

; Check that we do not use 32 temp vgprs, but rotate 3 vgprs only.
; 3 vgprs are needed to avoid wait states between writes.

; FIXME: We should not be using and temporary registers at all.
; At the moment we initialize an sgpr, then copy it via vgprs.

; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP2:v[0-9]+]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP3:v[0-9]+]]

; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP1:v[0-9]+]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP2]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP3]]

; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP1]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP2]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP3]]

; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP1]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP2]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP3]]

; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP1]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP2]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP3]]

; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP1]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP2]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP3]]

; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP1]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP2]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP3]]

; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP1]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP2]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP3]]

; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP1]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP2]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP3]]

; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP1]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP2]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP3]]

; Check that we do not copy agprs to vgprs and back inside the loop.

; GCN: [[LOOP:BB[0-9_]+]]:
; GCN-NOT: v_accvgpr
; GCN: v_mfma_f32_32x32x1f32
; GCN-NOT: v_accvgpr
; GCN: s_cbranch_scc1 [[LOOP]]

; Final result should be read only once after the loop.

; GCN-COUNT32: v_accvgpr_read_b32

define amdgpu_kernel void @test_mfma_loop_zeroinit(<32 x float> addrspace(1)* %arg) {
entry:
  br label %for.cond.preheader

for.cond.preheader:
  %phi = phi <32 x float> [ zeroinitializer, %entry ], [ %mai.1, %for.cond.preheader ]
  %c = phi i32 [ 0, %entry ], [ %inc, %for.cond.preheader ]
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %phi, i32 0, i32 0, i32 0)
  %inc = add nuw nsw i32 %c, 1
  %cc = icmp eq i32 %inc, 16
  br i1 %cc, label %exit, label %for.cond.preheader

exit:
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32, i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()
