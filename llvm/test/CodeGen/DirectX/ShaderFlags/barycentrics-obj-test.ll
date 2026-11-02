; RUN: llc -filetype=obj %s -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %s

; An unused SV_Barycentrics input still sets the Barycentrics feature flag.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

; CHECK: Name: SFI0
; CHECK: Barycentrics: true
; CHECK: Name: ISG1
; CHECK: Name: SV_Barycentrics
; CHECK: ExclusiveMask: 0
