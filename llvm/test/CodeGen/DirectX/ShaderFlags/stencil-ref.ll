; RUN: opt -disable-output -passes=print-dx-shader-flags %s 2>&1 | FileCheck %s

; Writing SV_StencilRef from a pixel shader requires the StencilRef feature,
; even if the output is never stored.

target triple = "dxil-pc-shadermodel6.0-library"

; CHECK: ; Shader Flags Value: [[FLAG:0x00000800]]
; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       PS Output Stencil Ref
; CHECK: Function stencil : [[FLAG]]
; CHECK: Function target : 0x00000000
define void @stencil() #0 {
entry:
  ret void
}
define void @target() #0 {
entry:
  ret void
}
attributes #0 = { "hlsl.shader"="pixel" }

!dx.semantic.signatures = !{!0, !3}
!0 = !{ptr @stencil, null, !1}
!1 = !{!2}
!2 = !{i32 0, !"SV_StencilRef", i32 5, i32 20, !5, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{ptr @target, null, !4}
!4 = !{!6}
!5 = !{i32 0}
!6 = !{i32 0, !"SV_Target", i32 9, i32 16, !5, i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
