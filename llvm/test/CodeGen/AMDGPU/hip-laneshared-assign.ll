; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1300 -max-vgprs-for-laneshared=8 -stop-after=amdgpu-assign-laneshared -verify-machineinstrs -o - %s | FileCheck -check-prefixes=VGPR8,CHECK %s
; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1300 -max-vgprs-for-laneshared=0 -stop-after=amdgpu-assign-laneshared -verify-machineinstrs -o - %s | FileCheck -check-prefixes=VGPR0,CHECK %s
; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1300 -max-vgprs-for-laneshared=4 -stop-after=amdgpu-assign-laneshared -verify-machineinstrs -o - %s | FileCheck -check-prefixes=VGPR4,CHECK %s

target datalayout = "A5"

@v1 = external local_unnamed_addr addrspace(10) global float, align 4
@vx = external local_unnamed_addr addrspace(10) global [7 x float], align 4
@v2 = external local_unnamed_addr addrspace(10) global float, align 4

; CHECK: @v1 = external local_unnamed_addr addrspace(10) global float, align 4, !absolute_symbol !0
; CHECK-NEXT: @vx = external local_unnamed_addr addrspace(10) global [7 x float], align 4, !absolute_symbol !1
; CHECK-NEXT: @v2 = external local_unnamed_addr addrspace(10) global float, align 4, !absolute_symbol !0
; VGPR8: !0 = !{i32 268435484, i32 268435485}
; VGPR8-NEXT: !1 = !{i32 268435456, i32 268435457}
; VGPR0: !0 = !{i32 28, i32 29}
; VGPR0-NEXT: !1 = !{i32 0, i32 1}
; VGPR4: !0 = !{i32 268435456, i32 268435457}
; VGPR4-NEXT: !1 = !{i32 0, i32 1}

define void @func1(float %a) {
    store float %a, ptr addrspace(10) getelementptr inbounds (i8, ptr addrspace(10) @vx, i32 12), align 4, !tbaa !4
    ret void
}

define float @func2() {
    %b = load float, ptr addrspace(10) getelementptr inbounds (i8, ptr addrspace(10) @vx, i32 20), align 4, !tbaa !4
    ret float %b
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define amdgpu_kernel void @_Kernel1() "amdgpu-wavegroup-enable" !reqd_work_group_size !8 {
entry:
  %0 = load float, ptr addrspace(10) @v1, align 4, !tbaa !4
  call void @func1(float %0)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define amdgpu_kernel void @_Kernel2() "amdgpu-wavegroup-enable" !reqd_work_group_size !8 {
entry:
  %1 = call float @func2()
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
!8 = !{i32 128, i32 1, i32 1}
