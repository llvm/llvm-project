; RUN: split-file %s %t
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/signed.ll 2>&1 | FileCheck %s --check-prefix=CHECK-SIGNED
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/cols.ll 2>&1 | FileCheck %s --check-prefix=CHECK-COLS
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/index.ll 2>&1 | FileCheck %s --check-prefix=CHECK-INDEX
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/kind.ll 2>&1 | FileCheck %s --check-prefix=CHECK-KIND
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/output.ll 2>&1 | FileCheck %s --check-prefix=CHECK-OUTPUT

; Each case is a complete module with one invalid SV_RenderTargetArrayIndex
; declaration.

;--- signed.ll
; CHECK-SIGNED: SV_RenderTargetArrayIndex requires a scalar 16- or 32-bit unsigned integer
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_RenderTargetArrayIndex", i32 4, i32 4, !3, i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- cols.ll
; CHECK-COLS: SV_RenderTargetArrayIndex requires a scalar 16- or 32-bit unsigned integer
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_RenderTargetArrayIndex", i32 5, i32 4, !3, i32 1, i32 1, i8 2, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- index.ll
; CHECK-INDEX: SV_RenderTargetArrayIndex requires semantic index 0
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_RenderTargetArrayIndex", i32 5, i32 4, !3, i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 1}

;--- kind.ll
; A vertex input with this name is an arbitrary semantic.
; CHECK-KIND: semantic name and kind disagree at this signature point
target triple = "dxil-pc-shadermodel6.0-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_RenderTargetArrayIndex", i32 5, i32 4, !3, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- output.ll
; CHECK-OUTPUT: invalid or unsupported semantic for this stage and signature category
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2}
!2 = !{i32 0, !"SV_RenderTargetArrayIndex", i32 5, i32 4, !3, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}
