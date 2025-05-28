; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1300 -max-vgprs-for-laneshared=0 -stop-after=finalize-isel -verify-machineinstrs -o - %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1300 -stop-after=finalize-isel -verify-machineinstrs -o - %s | FileCheck -check-prefix=VIDX %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1300 -verify-machineinstrs -o - %s | FileCheck -check-prefix=VGPR %s

; The 1st and 2nd run-line check can be generated using mir-check tool.
; The 3rd run-line check is added manually to check the min-vgpr restriction when a kernel has wmma.

target datalayout = "A5"

@a = external local_unnamed_addr addrspace(10) global <8 x i16>, align 16
@b = external local_unnamed_addr addrspace(10) global <8 x i16>, align 16
@out = external local_unnamed_addr addrspace(10) global <8 x float>, align 32

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)

define amdgpu_kernel void @_Z37test_amdgcn_wmma_f32_16x16x16_f16_w32v() "amdgpu-wavegroup-enable" !reqd_work_group_size !{i32 32, i32 12, i32 1} {
  ; CHECK-LABEL: name: _Z37test_amdgcn_wmma_f32_16x16x16_f16_w32v
  ; CHECK: bb.0.entry:
  ; CHECK-NEXT:   [[S_MOV_B32_:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 0
  ; CHECK-NEXT:   [[SCRATCH_LOAD_DWORDX4_SADDR:%[0-9]+]]:vreg_128 = SCRATCH_LOAD_DWORDX4_SADDR killed [[S_MOV_B32_]], 0, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (s128) from @a, align 2147483648, !tbaa !4, addrspace 10)
  ; CHECK-NEXT:   [[S_MOV_B32_1:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 16
  ; CHECK-NEXT:   [[SCRATCH_LOAD_DWORDX4_SADDR1:%[0-9]+]]:vreg_128 = SCRATCH_LOAD_DWORDX4_SADDR killed [[S_MOV_B32_1]], 0, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (s128) from @b, !tbaa !4, addrspace 10)
  ; CHECK-NEXT:   early-clobber %10:vreg_256 = contract V_WMMA_F32_16X16X16_F16_threeaddr killed [[SCRATCH_LOAD_DWORDX4_SADDR]], killed [[SCRATCH_LOAD_DWORDX4_SADDR1]], 0, 0, 0, 0, implicit $exec
  ; CHECK-NEXT:   [[COPY:%[0-9]+]]:vgpr_32 = COPY %10.sub7
  ; CHECK-NEXT:   [[COPY1:%[0-9]+]]:vgpr_32 = COPY %10.sub6
  ; CHECK-NEXT:   [[COPY2:%[0-9]+]]:vgpr_32 = COPY %10.sub5
  ; CHECK-NEXT:   [[COPY3:%[0-9]+]]:vgpr_32 = COPY %10.sub4
  ; CHECK-NEXT:   [[DEF:%[0-9]+]]:sgpr_32 = IMPLICIT_DEF
  ; CHECK-NEXT:   [[DEF1:%[0-9]+]]:sgpr_32 = IMPLICIT_DEF
  ; CHECK-NEXT:   [[DEF2:%[0-9]+]]:sgpr_32 = IMPLICIT_DEF
  ; CHECK-NEXT:   [[DEF3:%[0-9]+]]:sgpr_32 = IMPLICIT_DEF
  ; CHECK-NEXT:   [[REG_SEQUENCE:%[0-9]+]]:vreg_128 = REG_SEQUENCE [[COPY3]], %subreg.sub0, [[COPY2]], %subreg.sub1, [[COPY1]], %subreg.sub2, [[COPY]], %subreg.sub3
  ; CHECK-NEXT:   [[S_MOV_B32_2:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 48
  ; CHECK-NEXT:   [[COPY4:%[0-9]+]]:vreg_128 = COPY [[REG_SEQUENCE]]
  ; CHECK-NEXT:   SCRATCH_STORE_DWORDX4_SADDR killed [[COPY4]], killed [[S_MOV_B32_2]], 0, 0, implicit $exec, implicit $flat_scr :: (store (s128) into @out + 16, addrspace 10)
  ; CHECK-NEXT:   [[COPY5:%[0-9]+]]:vgpr_32 = COPY %10.sub3
  ; CHECK-NEXT:   [[COPY6:%[0-9]+]]:vgpr_32 = COPY %10.sub2
  ; CHECK-NEXT:   [[COPY7:%[0-9]+]]:vgpr_32 = COPY %10.sub1
  ; CHECK-NEXT:   [[COPY8:%[0-9]+]]:vgpr_32 = COPY %10.sub0
  ; CHECK-NEXT:   [[DEF4:%[0-9]+]]:sgpr_32 = IMPLICIT_DEF
  ; CHECK-NEXT:   [[DEF5:%[0-9]+]]:sgpr_32 = IMPLICIT_DEF
  ; CHECK-NEXT:   [[DEF6:%[0-9]+]]:sgpr_32 = IMPLICIT_DEF
  ; CHECK-NEXT:   [[DEF7:%[0-9]+]]:sgpr_32 = IMPLICIT_DEF
  ; CHECK-NEXT:   [[REG_SEQUENCE1:%[0-9]+]]:vreg_128 = REG_SEQUENCE [[COPY8]], %subreg.sub0, [[COPY7]], %subreg.sub1, [[COPY6]], %subreg.sub2, [[COPY5]], %subreg.sub3
  ; CHECK-NEXT:   [[S_MOV_B32_3:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 32
  ; CHECK-NEXT:   [[COPY9:%[0-9]+]]:vreg_128 = COPY [[REG_SEQUENCE1]]
  ; CHECK-NEXT:   SCRATCH_STORE_DWORDX4_SADDR killed [[COPY9]], killed [[S_MOV_B32_3]], 0, 0, implicit $exec, implicit $flat_scr :: (store (s128) into @out, align 32, addrspace 10)
  ; CHECK-NEXT:   S_ENDPGM 0
  ;
  ; VIDX-LABEL: name: _Z37test_amdgcn_wmma_f32_16x16x16_f16_w32v
  ; VIDX: bb.0.entry:
  ; VIDX-NEXT:   [[S_MOV_B32_:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 0
  ; VIDX-NEXT:   [[S_LSHR_B32_:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[S_MOV_B32_]], 2, implicit-def dead $scc
  ; VIDX-NEXT:   [[V_LOAD_IDX:%[0-9]+]]:vreg_128 = V_LOAD_IDX [[S_LSHR_B32_]], 0, implicit $exec :: (dereferenceable load (s128) from @a, align 268435456, !tbaa !4, addrspace 10)
  ; VIDX-NEXT:   [[S_MOV_B32_1:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 16
  ; VIDX-NEXT:   [[S_LSHR_B32_1:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[S_MOV_B32_1]], 2, implicit-def dead $scc
  ; VIDX-NEXT:   [[V_LOAD_IDX1:%[0-9]+]]:vreg_128 = V_LOAD_IDX [[S_LSHR_B32_1]], 0, implicit $exec :: (dereferenceable load (s128) from @b, !tbaa !4, addrspace 10)
  ; VIDX-NEXT:   early-clobber %10:vreg_256 = contract V_WMMA_F32_16X16X16_F16_threeaddr killed [[V_LOAD_IDX]], killed [[V_LOAD_IDX1]], 0, 0, 0, 0, implicit $exec
  ; VIDX-NEXT:   [[S_MOV_B32_2:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 32
  ; VIDX-NEXT:   [[S_LSHR_B32_2:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[S_MOV_B32_2]], 2, implicit-def dead $scc
  ; VIDX-NEXT:   V_STORE_IDX %10, [[S_LSHR_B32_2]], 0, implicit $exec :: (store (s256) into @out, !tbaa !4, addrspace 10)
  ; VIDX-NEXT:   S_ENDPGM 0
entry:
  %0 = load <8 x half>, ptr addrspace(10) @a, align 16, !tbaa !4
  %1 = load <8 x half>, ptr addrspace(10) @b, align 16, !tbaa !4
  %2 = tail call contract <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.clamp(<8 x half> %0, <8 x half> %1, <8 x float> zeroinitializer, i1 0)
  store <8 x float> %2, ptr addrspace(10) @out, align 32, !tbaa !4

  ret void
}

; VGPR: ; NumVgprs: 16
; VGPR: ; NumVGPRsForWavesPerEU: 32

!4 = !{!5, !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
