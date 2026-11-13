; RUN: split-file %s %t
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/bool.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BOOL
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/float.ll 2>&1 | FileCheck %s --check-prefix=CHECK-FLOAT
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/cols.ll 2>&1 | FileCheck %s --check-prefix=CHECK-COLS
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/index.ll 2>&1 | FileCheck %s --check-prefix=CHECK-INDEX
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/linear.ll 2>&1 | FileCheck %s --check-prefix=CHECK-LINEAR
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/unallocated.ll 2>&1 | FileCheck %s --check-prefix=CHECK-UNALLOCATED
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/vertex.ll 2>&1 | FileCheck %s --check-prefix=CHECK-VERTEX
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/output.ll 2>&1 | FileCheck %s --check-prefix=CHECK-OUTPUT

; Each case is a complete module with one invalid SV_IsFrontFace declaration.
; The only valid component types are I32 (4) and U32 (5). Clang carries a bool
; as U32.

;--- bool.ll
; The component type is 1 (I1). Clang never emits it for SV_IsFrontFace, and it
; is not a valid type for this semantic.
; CHECK-BOOL: SV_IsFrontFace requires a scalar 32-bit integer
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_IsFrontFace", i32 1, i32 13, !3, i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- float.ll
; The component type is 9 (F32).
; CHECK-FLOAT: SV_IsFrontFace requires a scalar 32-bit integer
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_IsFrontFace", i32 9, i32 13, !3, i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- cols.ll
; CHECK-COLS: SV_IsFrontFace requires a scalar 32-bit integer
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_IsFrontFace", i32 5, i32 13, !3, i32 1, i32 1, i8 2, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- index.ll
; CHECK-INDEX: SV_IsFrontFace requires semantic index 0
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_IsFrontFace", i32 5, i32 13, !3, i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 1}

;--- linear.ll
; CHECK-LINEAR: integer pixel inputs require constant interpolation
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_IsFrontFace", i32 5, i32 13, !3, i32 2, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- unallocated.ll
; CHECK-UNALLOCATED: signature element 0 ('SV_IsFrontFace') has no allocated location
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_IsFrontFace", i32 5, i32 13, !3, i32 1, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- vertex.ll
; CHECK-VERTEX: invalid or unsupported semantic for this stage and signature category
target triple = "dxil-pc-shadermodel6.0-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_IsFrontFace", i32 5, i32 13, !3, i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- output.ll
; SV_IsFrontFace is only a pixel input, so a pixel output is rejected.
; CHECK-OUTPUT: invalid or unsupported semantic for this stage and signature category
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2}
!2 = !{i32 0, !"SV_IsFrontFace", i32 5, i32 13, !3, i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}
