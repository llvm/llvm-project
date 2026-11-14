; RUN: opt -S -dxil-translate-metadata %s | FileCheck %s

; SV_ViewID is not part of the signature. Reading it sets the ViewID shader
; flag, and every written output component is conservatively treated as
; depending on it in dx.viewIdState. The container parts are covered by
; ContainerData/SemanticSignatures-view-id.ll and the dx.op.viewID lowering by
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

; The state is the input and output scalar counts, the outputs depending on
; the view ID (components 0 and 2), then one output mask per input scalar.
; CHECK-DAG: !dx.viewIdState = !{![[STATE:[0-9]+]]}
; CHECK-DAG: ![[STATE]] = !{[7 x i32] [i32 4, i32 4, i32 5, i32 5, i32 0, i32 0, i32 0]}

; The entry properties carry the ViewID shader flag (0x00200000).
; CHECK-DAG: !{i32 0, i64 2097152}
