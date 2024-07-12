; RUN: llc -global-isel -mtriple=amdgcn-- -mcpu=gfx1300 -amdgpu-promote-lane-shared=false -stop-after=finalize-isel -verify-machineinstrs -o - %s | FileCheck %s
; RUN: llc -global-isel -mtriple=amdgcn-- -mcpu=gfx1300 -stop-after=finalize-isel -verify-machineinstrs -o - %s | FileCheck -check-prefix=VIDX %s
target datalayout = "A5"

@a = external local_unnamed_addr addrspace(10) global <8 x i16>, align 16
@b = external local_unnamed_addr addrspace(10) global <8 x i16>, align 16
@out = external local_unnamed_addr addrspace(10) global <8 x float>, align 32

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)

define dso_local amdgpu_kernel void @_Z37test_amdgcn_wmma_f32_16x16x16_f16_w32v() local_unnamed_addr {
entry:
  %0 = load <8 x half>, ptr addrspace(10) @a, align 16, !tbaa !4
  %1 = load <8 x half>, ptr addrspace(10) @b, align 16, !tbaa !4
; CHECK: [[SADDR1:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 48
; CHECK-NEXT: [[MATA:%[0-9]+]]:vreg_128_align2 = SCRATCH_LOAD_DWORDX4_SADDR [[SADDR1]], 0, 0, implicit $exec, implicit $flat_scr
; CHECK: [[SADDR2:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 32
; CHECK-NEXT: [[MATB:%[0-9]+]]:vreg_128_align2 = SCRATCH_LOAD_DWORDX4_SADDR [[SADDR2]], 0, 0, implicit $exec, implicit $flat_scr
; CHECK: early-clobber [[MATD:%[0-9]+]]:vreg_256_align2 = contract V_WMMA_F32_16X16X16_F16_w32_threeaddr 8, [[MATA]], 8, [[MATB]], 8, 0, 0, 0, implicit $exec
  %2 = tail call contract <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %0, <8 x half> %1, <8 x float> zeroinitializer)
  store <8 x float> %2, ptr addrspace(10) @out, align 32, !tbaa !4
; CHECK: SCRATCH_STORE_DWORDX4_SADDR
; CHECK: SCRATCH_STORE_DWORDX4_SADDR
; VIDX: [[OFF1:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 48
; VIDX-NEXT: [[SFR1:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[OFF1]], 2, implicit-def dead $scc
; VIDX-NEXT: [[MATA:%[0-9]+]]:vreg_128_align2 = V_LOAD_IDX [[SFR1]], 0, 1, implicit $exec
; VIDX-NEXT: [[OFF2:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 32
; VIDX-NEXT: [[SFR2:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[OFF2]], 2, implicit-def dead $scc
; VIDX-NEXT: [[MATB:%[0-9]+]]:vreg_128_align2 = V_LOAD_IDX [[SFR2]], 0, 1, implicit $exec
; VIDX-NEXT: early-clobber [[MATD:%[0-9]+]]:vreg_256_align2 = contract V_WMMA_F32_16X16X16_F16_w32_threeaddr 8, [[MATA]], 8, [[MATB]], 8, 0, 0, 0, implicit $exec
; VIDX-NEXT: [[OFF3:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 0
; VIDX-NEXT: [[SFR3:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[OFF3]], 2, implicit-def dead $scc
; VIDX-NEXT: V_STORE_IDX [[MATD]], [[SFR3]], 0, 1, implicit $exec
  ret void
}

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half>, <8 x half>, <8 x float>) #1

attributes #1 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!4 = !{!5, !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}