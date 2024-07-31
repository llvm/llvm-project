; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1300 -verify-machineinstrs -stop-after=finalize-isel -o - %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1300 -verify-machineinstrs -stop-after=amdgpu-idx-reg-alloc -o - %s | FileCheck -check-prefix=SETIDX %s

target datalayout = "A5"

@v1 = external local_unnamed_addr addrspace(10) global float, align 4
@vx = external local_unnamed_addr addrspace(10) global [7 x float], align 4
@v2 = external local_unnamed_addr addrspace(10) global float, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define dso_local amdgpu_kernel void @_Z3foov() local_unnamed_addr {
entry:
; CHECK:      [[T1:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 32
; CHECK-NEXT: [[T2:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[T1]], 2, implicit-def dead $scc
; CHECK-NEXT: [[T3:%[0-9]+]]:vgpr_32 = V_LOAD_IDX [[T2]], 0, implicit $exec
; CHECK:      [[T4:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 16
; CHECK-NEXT: [[T5:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[T4]], 2, implicit-def dead $scc
; CHECK-NEXT: V_STORE_IDX [[T3]], [[T5]], 0, implicit $exec
; CHECK:      [[T6:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 24
; CHECK-NEXT: [[T7:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[T6]], 2, implicit-def dead $scc
; CHECK-NEXT: [[T8:%[0-9]+]]:vgpr_32 = V_LOAD_IDX [[T7]], 0, implicit $exec
; CHECK:      [[T9:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 0
; CHECK-NEXT: [[T10:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[T9]], 2, implicit-def dead $scc
; CHECK-NEXT: V_STORE_IDX [[T8]], [[T10]], 0, implicit $exec
; SETIDX:     [[ZERO:%[0-9]+]]:sgpr_32 = S_MOV_B32 0
; SETIDX-NEXT: $idx1 = S_SET_GPR_IDX_U32 [[ZERO]]
; SETIDX-NEXT: [[DAT0:%[0-9]+]]:vgpr_32 = V_LOAD_IDX $idx1, 8, implicit $exec
; SETIDX-NEXT: V_STORE_IDX [[DAT0]], $idx1, 4, implicit $exec
; SETIDX-NEXT: [[DAT1:%[0-9]+]]:vgpr_32 = V_LOAD_IDX $idx1, 6, implicit $exec
; SETIDX-NEXT: V_STORE_IDX [[DAT1]], $idx1, 0, implicit $exec
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
