; RUN: llc -global-isel -mtriple=amdgcn-- -mcpu=gfx1300 -amdgpu-promote-lane-shared=false -stop-after=finalize-isel -verify-machineinstrs -o - %s | FileCheck %s
; RUN: llc -global-isel -mtriple=amdgcn-- -mcpu=gfx1300 -stop-after=finalize-isel -o - %s | FileCheck -check-prefix=VIDX %s
target datalayout = "A5"

@weights = external local_unnamed_addr addrspace(10) global <9 x i32>, align 64
@col_center = external local_unnamed_addr addrspace(10) global <3 x i32>, align 16
@col_left = external local_unnamed_addr addrspace(10) global <3 x i32>, align 16
@col_right = external local_unnamed_addr addrspace(10) global <3 x i32>, align 16
@out = external local_unnamed_addr addrspace(10) global <8 x i16>, align 16

; Function Attrs: convergent mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none)
define dso_local amdgpu_kernel void @_Z36test_amdgcn_convolve_f16_fp8_3x3_4x4v() local_unnamed_addr {
entry:
  %vec30 = load <3 x i32>, ptr addrspace(10) @col_center, align 16
  %vec31 = load <3 x i32>, ptr addrspace(10) @col_left, align 16
  %vec32 = load <3 x i32>, ptr addrspace(10) @col_right, align 16
  %wei = load <9 x i32>, ptr addrspace(10) @weights, align 64
; CHECK: [[OFFS:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 64
; CHECK-NEXT: [[SUB0:%[0-9]+]]:vreg_96_align2 = SCRATCH_LOAD_DWORDX3_SADDR [[OFFS]], 0, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (<3 x s32>) from @weights, align 64, addrspace 10)
; CHECK-NEXT: [[SUB3:%[0-9]+]]:vreg_96_align2 = SCRATCH_LOAD_DWORDX3_SADDR [[OFFS]], 12, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (<3 x s32>) from @weights + 12, align 4, basealign 64, addrspace 10)
; CHECK-NEXT: [[SUB6:%[0-9]+]]:vreg_96_align2 = SCRATCH_LOAD_DWORDX3_SADDR [[OFFS]], 24, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (<3 x s32>) from @weights + 24, align 8, basealign 64, addrspace 10)
; CHECK-NEXT: [[SEQ:%[0-9]+]]:vreg_288_align2 = REG_SEQUENCE [[SUB0]], %subreg.sub0_sub1_sub2, [[SUB3]], %subreg.sub3_sub4_sub5, [[SUB6]], %subreg.sub6_sub7_sub8
; VIDX: [[OFFS:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 16
; VIDX-NEXT: [[VIDX:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[OFFS]], 2, implicit-def dead $scc
; VIDX-NEXT: [[VDAT:%[0-9]+]]:vreg_288_align2 = V_LOAD_IDX [[VIDX]], 0, implicit $exec
; VIDX-DAG:  [[CONV:%[0-9]+]]:vreg_128_align2 = contract V_CONVOLVE_F16_FP8_3x3_4x4 {{%[0-9]+}}, [[VDAT]],
  %0 = tail call contract <8 x half> @llvm.amdgcn.convolve.f16.fp8.3x3.v8f16.v8f16.v9i32.v3i32(<8 x half> zeroinitializer, <9 x i32> %wei, <3 x i32> %vec30, <3 x i32> %vec31, <3 x i32> %vec32, i32 42, i1 true)
  store <8 x half> %0, ptr addrspace(10) @out, align 16, !tbaa !4
  ret void
}

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <8 x half> @llvm.amdgcn.convolve.f16.fp8.3x3.v8f16.v8f16.v9i32.v3i32(<8 x half>, <9 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i32 immarg, i1 immarg) #1

!4 = !{!5, !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
