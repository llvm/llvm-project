; RUN: split-file %s %t
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/signed.ll 2>&1 | FileCheck %s --check-prefix=CHECK-SIGNED
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/cols.ll 2>&1 | FileCheck %s --check-prefix=CHECK-COLS
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/index.ll 2>&1 | FileCheck %s --check-prefix=CHECK-INDEX
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/allocated.ll 2>&1 | FileCheck %s --check-prefix=CHECK-ALLOCATED
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/input.ll 2>&1 | FileCheck %s --check-prefix=CHECK-INPUT
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/vertex.ll 2>&1 | FileCheck %s --check-prefix=CHECK-VERTEX

; Each case is a complete module with one invalid SV_StencilRef declaration.

;--- signed.ll
; CHECK-SIGNED: SV_StencilRef requires a scalar 16- or 32-bit unsigned integer
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2}
!2 = !{i32 0, !"SV_StencilRef", i32 4, i32 20, !3, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- cols.ll
; CHECK-COLS: SV_StencilRef requires a scalar 16- or 32-bit unsigned integer
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2}
!2 = !{i32 0, !"SV_StencilRef", i32 5, i32 20, !3, i32 0, i32 1, i8 2, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- index.ll
; CHECK-INDEX: SV_StencilRef requires semantic index 0
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2}
!2 = !{i32 0, !"SV_StencilRef", i32 5, i32 20, !3, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 1}

;--- allocated.ll
; CHECK-ALLOCATED: SV_StencilRef must not have an allocated location
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2}
!2 = !{i32 0, !"SV_StencilRef", i32 5, i32 20, !3, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- vertex.ll
; CHECK-VERTEX: invalid or unsupported semantic for this stage and signature category
target triple = "dxil-pc-shadermodel6.0-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2}
!2 = !{i32 0, !"SV_StencilRef", i32 5, i32 20, !3, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- input.ll
; CHECK-INPUT: invalid or unsupported semantic for this stage and signature category
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_StencilRef", i32 5, i32 20, !3, i32 1, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}
