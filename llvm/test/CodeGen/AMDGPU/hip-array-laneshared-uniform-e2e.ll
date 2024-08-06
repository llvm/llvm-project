; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1300 -verify-machineinstrs -o - %s | FileCheck %s

target datalayout = "A5"

@exchange = external local_unnamed_addr addrspace(10) global [70 x float], align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none)
define dso_local amdgpu_kernel void @_Z3foov() local_unnamed_addr {
entry:
; CHECK: s_getreg_b32 s33, hwreg(HW_REG_WAVE_GROUP_INFO, 16, 3)
; CHECK: s_mul_i32 s33, s33, [[VGPRsForWavesPerEU:[0-9]+]]
; CHECK: s_add_co_i32 s33, s33, 0x46
; CHECK: s_set_gpr_idx_u32 idx0, s33
; CHECK: s_getreg_b32 s33, hwreg(HW_REG_WAVE_GROUP_INFO, 16, 3)
; CHECK: s_mul_i32 s33, s33, 0
; CHECK: s_add_co_i32 s33, s33, 0
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.024 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %promotealloca23 = phi <10 x float> [ undef, %entry ], [ %1, %for.body ]
  %arrayidx = getelementptr inbounds [70 x float], ptr addrspace(10) @exchange, i32 0, i32 %i.024
; CHECK: s_set_gpr_idx_u32 idx1,
; CHECK: s_set_vgpr_frames 1                     ;  vsrc0_idx=1 vsrc1_idx=0 vsrc2_idx=0 vdst_idx=0 vsrc0_msb=0 vsrc1_msb=0 vsrc2_msb=0 vdst_msb=0
; CHECK: v_mov_b32_e32 v{{[0-9]+}}, v0
; CHECK: s_set_vgpr_frames 0
; CHECK: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
  %0 = load float, ptr addrspace(10) %arrayidx, align 4, !tbaa !4
  %rem = urem i32 %i.024, 10
  %idxprom1 = zext nneg i32 %rem to i64
  %1 = insertelement <10 x float> %promotealloca23, float %0, i64 %idxprom1
  %inc = add nuw nsw i32 %i.024, 1
  %exitcond.not = icmp eq i32 %inc, 70
  br i1 %exitcond.not, label %for.body8, label %for.body, !llvm.loop !8

for.cond.cleanup6:                                ; preds = %for.body8
  ret void

for.body8:                                        ; preds = %for.body, %for.body8
  %i3.025 = phi i32 [ %inc15, %for.body8 ], [ 0, %for.body ]
  %rem9 = urem i32 %i3.025, 10
  %idxprom10 = zext nneg i32 %rem9 to i64
  %2 = extractelement <10 x float> %1, i64 %idxprom10
  %arrayidx13 = getelementptr inbounds [70 x float], ptr addrspace(10) @exchange, i32 0, i32 %i3.025
; CHECK: s_set_gpr_idx_u32 idx1,
; CHECK: s_set_vgpr_frames 64                    ;  vsrc0_idx=0 vsrc1_idx=0 vsrc2_idx=0 vdst_idx=1 vsrc0_msb=0 vsrc1_msb=0 vsrc2_msb=0 vdst_msb=0
; CHECK: v_mov_b32_e32 v0, v{{[0-9]+}}
; CHECK: s_set_vgpr_frames 0
  store float %2, ptr addrspace(10) %arrayidx13, align 4, !tbaa !4
  %inc15 = add nuw nsw i32 %i3.025, 1
  %exitcond26.not = icmp eq i32 %inc15, 70
  br i1 %exitcond26.not, label %for.cond.cleanup6, label %for.body8, !llvm.loop !11
}

; CHECK: NumVGPRsForWavesPerEU: [[VGPRsForWavesPerEU]]

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = distinct !{!8, !9, !10}
!9 = !{!"llvm.loop.mustprogress"}
!10 = !{!"llvm.loop.unroll.disable"}
!11 = distinct !{!11, !9, !10}
