; RUN: opt -S -passes=spirv-remove-unused-resources %s -o - | FileCheck %s

; Check that, when selecting frag_ep1 as the entry point, we don't remove the global
; since the dead resource removal pass is only intended for handles.
; Original HLSL:
;
; static float saved;
;
; float frag_ep1(float value : TEXCOORD0) : SV_Target {
;   saved = value;
;   return value;
; }
;
; float frag_ep2() : SV_Target {
;   return saved;
; }

; CHECK: @_ZL5saved = internal {{.*}}addrspace(10) global float
; CHECK-LABEL: define void @frag_ep1()
; CHECK: %[[VALUE:[0-9]+]] = load float, ptr addrspace(7) @TEXCOORD0
; CHECK-NEXT: store float %[[VALUE]], ptr addrspace(10) @_ZL5saved
; CHECK: ret void

target datalayout = "e-ve-i64:64-n8:16:32:64-G10"
target triple = "spirv1.6-unknown-vulkan1.3-pixel"

@TEXCOORD0 = external hidden thread_local local_unnamed_addr addrspace(7) externally_initialized constant float, !spirv.Decorations !5
@SV_Target0 = external hidden thread_local local_unnamed_addr addrspace(8) global float, !spirv.Decorations !5
@_ZL5saved = internal unnamed_addr addrspace(10) global float 0.000000e+00, align 4

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none, target_mem: none)
define void @frag_ep1() local_unnamed_addr #0 {
entry:
  %0 = load float, ptr addrspace(7) @TEXCOORD0, align 4
  store float %0, ptr addrspace(10) @_ZL5saved, align 4, !tbaa !18
  store float %0, ptr addrspace(8) @SV_Target0, align 4
  ret void
}

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none, target_mem: none) "hlsl.shader"="pixel" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.errno.tbaa = !{!10}

!5 = !{!6}
!6 = !{i32 30, i32 0}
!10 = !{!11, !13, i64 0}
!11 = !{!"__libc_errno", !13, i64 0}
!13 = !{!"int", !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C++ TBAA"}
!18 = !{!19, !19, i64 0}
!19 = !{!"float", !15, i64 0}
