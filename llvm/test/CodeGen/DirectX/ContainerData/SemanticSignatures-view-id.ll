; RUN: llc -filetype=obj %s -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %s

; SV_ViewID is not part of the signature. Reading it sets ViewID in SFI0 and
; UsesViewID in PSV0, and every written output component is conservatively
; treated as depending on it in PSV0. The dx.viewIdState metadata is covered by
; Metadata/semantic-signatures-view-id.ll and the dx.op.viewID lowering by
; view_id.ll.
target triple = "dxil-pc-shadermodel6.1-pixel"

define void @main() #0 {
  %a = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  %view = call i32 @llvm.dx.view.id()
  %f = uitofp i32 %view to float
  %result = fadd float %a, %f
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %result)
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 2, float %result)
  ret void
}
attributes #0 = { "hlsl.shader"="pixel" }

!dx.valver = !{!10}
!10 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3}
!2 = !{!6}
!3 = !{i32 0, !"A", i32 9, i32 0, !7, i32 2, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!6 = !{i32 0, !"SV_Target", i32 9, i32 16, !7, i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
!7 = !{i32 0}

; CHECK: Name: SFI0
; CHECK: ViewID: true
; CHECK: Name: PSV0
; CHECK: UsesViewID: 1
; Outputs depending on the view ID: components 0 and 2.
; CHECK:      OutputVectorMasks:
; CHECK-NEXT:   - [ 0x5 ]
; CHECK:      InputOutputMap:
; CHECK-NEXT:   - [ 0x5, 0x0, 0x0, 0x0 ]
