; RUN: llc -global-isel -mtriple=amdgcn-- -mcpu=gfx1300 -verify-machineinstrs -stop-after=finalize-isel -o - %s | FileCheck %s

target datalayout = "A5"

@v1 = external local_unnamed_addr addrspace(10) global float, align 4
@vx = external local_unnamed_addr addrspace(10) global [7 x float], align 4
@v2 = external local_unnamed_addr addrspace(10) global float, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define dso_local amdgpu_kernel void @_Z3foov() local_unnamed_addr {
entry:
; CHECK:      [[VX:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 8
; CHECK-NEXT: [[V1:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 4
; CHECK-NEXT: [[T2:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[V1]], 2, implicit-def dead $scc
; CHECK-NEXT: [[T3:%[0-9]+]]:vgpr_32 = V_LOAD_IDX [[T2]], 0, 1, implicit $exec
; CHECK-NEXT: [[T5:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[VX]], 2, implicit-def dead $scc
; CHECK-NEXT: V_STORE_IDX [[T3]], [[T5]], 3, 1, implicit $exec
; CHECK-NEXT: [[T7:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[VX]], 2, implicit-def dead $scc
; CHECK-NEXT: [[T8:%[0-9]+]]:vgpr_32 = V_LOAD_IDX [[T7]], 5, 1, implicit $exec
; CHECK-NEXT: [[V2:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 0
; CHECK-NEXT: [[T10:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[V2]], 2, implicit-def dead $scc
; CHECK-NEXT: V_STORE_IDX [[T8]], [[T10]], 0, 1, implicit $exec
  %0 = load float, ptr addrspace(10) @v1, align 4, !tbaa !4
  store float %0, ptr addrspace(10) getelementptr inbounds (i8, ptr addrspace(10) @vx, i32 12), align 4, !tbaa !4
  %1 = load float, ptr addrspace(10) getelementptr inbounds (i8, ptr addrspace(10) @vx, i32 20), align 4, !tbaa !4
  store float %1, ptr addrspace(10) @v2, align 4, !tbaa !4
  ret void
}

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
