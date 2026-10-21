; RUN: opt -disable-output -passes=print-dx-shader-flags %s 2>&1 | FileCheck %s

; Writing SV_RenderTargetArrayIndex from a vertex shader requires the
; ViewportAndRTArrayIndexFromAnyShaderFeedingRasterizer feature. Reading it in
; a pixel shader does not.

target triple = "dxil-pc-shadermodel6.0-library"

; CHECK: ; Shader Flags Value: [[FLAG:0x00000200]]
; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       SV_RenderTargetArrayIndex or SV_ViewportArrayIndex from any shader feeding rasterizer
; CHECK: Function vertex : [[FLAG]]
; CHECK: Function pixel : 0x00000000
define void @vertex() #0 {
entry:
  ret void
}
define void @pixel() #1 {
entry:
  ret void
}
attributes #0 = { "hlsl.shader"="vertex" }
attributes #1 = { "hlsl.shader"="pixel" }

!dx.semantic.signatures = !{!0, !3}
!0 = !{ptr @vertex, null, !1}
!1 = !{!2}
!2 = !{i32 0, !"SV_RenderTargetArrayIndex", i32 5, i32 4, !5, i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{ptr @pixel, !4, null}
!4 = !{!2}
!5 = !{i32 0}
