; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1300 -verify-machineinstrs -stop-after=finalize-isel -o - %s | FileCheck %s

target datalayout = "A5"

@v1 = external local_unnamed_addr addrspace(10) global float, align 4
@vx = external local_unnamed_addr addrspace(10) global [7 x float], align 4
@v2 = external local_unnamed_addr addrspace(10) global float, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define dso_local amdgpu_kernel void @_Z3foov() local_unnamed_addr #0 {
entry:
; CHECK:      [[T1:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 32
; CHECK-NEXT: [[T2:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[T1]], 2, implicit-def $scc
; CHECK-NEXT: $idx1 = S_SET_GPR_IDX_U32 [[T2]]
; CHECK-NEXT: [[T3:%[0-9]+]]:vgpr_32 = V_LOAD_IDX $idx1, 0, 1, implicit $exec
; CHECK:      [[T4:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 16
; CHECK-NEXT: [[T5:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[T4]], 2, implicit-def $scc
; CHECK-NEXT: $idx1 = S_SET_GPR_IDX_U32 [[T5]]
; CHECK-NEXT: V_STORE_IDX [[T3]], $idx1, 0, 1, implicit $exec
; CHECK:      [[T6:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 24
; CHECK-NEXT: [[T7:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[T6]], 2, implicit-def $scc
; CHECK-NEXT: $idx1 = S_SET_GPR_IDX_U32 [[T7]]
; CHECK-NEXT: [[T8:%[0-9]+]]:vgpr_32 = V_LOAD_IDX $idx1, 0, 1, implicit $exec
; CHECK:      [[T9:%[0-9]+]]:sreg_32_xexec_hi = S_MOV_B32 0
; CHECK-NEXT: [[T10:%[0-9]+]]:sreg_32_xexec_hi = S_LSHR_B32 [[T9]], 2, implicit-def $scc
; CHECK-NEXT: $idx1 = S_SET_GPR_IDX_U32 [[T10]]
; CHECK-NEXT: V_STORE_IDX [[T8]], $idx1, 0, 1, implicit $exec
  %0 = load float, ptr addrspace(10) @v1, align 4, !tbaa !4
  store float %0, ptr addrspace(10) getelementptr inbounds (i8, ptr addrspace(10) @vx, i32 12), align 4, !tbaa !4
  %1 = load float, ptr addrspace(10) getelementptr inbounds (i8, ptr addrspace(10) @vx, i32 20), align 4, !tbaa !4
  store float %1, ptr addrspace(10) @v2, align 4, !tbaa !4
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) "amdgpu-flat-work-group-size"="1,1024" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1300" "target-features"="+16-bit-insts,+ashr-pk-insts,+atomic-buffer-pk-add-bf16-inst,+bf16-cvt-insts,+bf16-pk-insts,+bf16-trans-insts,+bitop3-insts,+ci-insts,+dl-insts,+dot7-insts,+dot8-insts,+dpp,+f16bf16-to-fp6bf6-cvt-scale-insts,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-10-insts,+gfx12-insts,+gfx13-insts,+gfx8-insts,+gfx9-insts,+permlane16-swap,+prng-inst,+tanh-insts,+wavefrontsize32" "uniform-work-group-size"="true" }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}