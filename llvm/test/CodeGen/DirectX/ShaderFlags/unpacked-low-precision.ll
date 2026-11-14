; RUN: opt -disable-output -passes=print-dx-shader-flags %s 2>&1 | FileCheck %s

; Unallocated (NotPacked) elements do not occupy signature registers, so only
; the instructions that read them contribute type requirements, as in DXC. A
; 16-bit SV_SampleIndex is read with the 32-bit dx.op.sampleIndex, and an
; unused 16-bit SV_Barycentrics has no loadInput. Allocated 16-bit signature
; elements still require low-precision support, including when they share a
; signature with an unallocated 16-bit element.

target triple = "dxil-pc-shadermodel6.2-library"

; CHECK: ; Shader Flags Value: 0x00400020
; CHECK: Function sample_index : 0x00000000
; CHECK: Function barycentrics : 0x00400000
; CHECK: Function arbitrary : 0x00000020
; CHECK: Function sample_index_and_arbitrary : 0x00000020
define void @sample_index() #0 {
entry:
  %0 = call i32 @llvm.dx.sample.index()
  ret void
}
define void @barycentrics() #0 {
entry:
  ret void
}
define void @arbitrary() #0 {
entry:
  ret void
}
define void @sample_index_and_arbitrary() #0 {
entry:
  %0 = call i32 @llvm.dx.sample.index()
  ret void
}
attributes #0 = { "hlsl.shader"="pixel" }

!dx.semantic.signatures = !{!0, !4, !7, !10}
!0 = !{ptr @sample_index, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_SampleIndex", i32 3, i32 12, !3, i32 1, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}
!4 = !{ptr @barycentrics, !5, null}
!5 = !{!6}
!6 = !{i32 0, !"SV_Barycentrics", i32 8, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!7 = !{ptr @arbitrary, !8, null}
!8 = !{!9}
!9 = !{i32 0, !"A", i32 3, i32 0, !3, i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!10 = !{ptr @sample_index_and_arbitrary, !11, null}
!11 = !{!2, !12}
!12 = !{i32 1, !"A", i32 3, i32 0, !3, i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
