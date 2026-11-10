; RUN: split-file %s %t
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/integer.ll 2>&1 | FileCheck %s --check-prefix=CHECK-INTEGER
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/cols.ll 2>&1 | FileCheck %s --check-prefix=CHECK-COLS
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/rows.ll 2>&1 | FileCheck %s --check-prefix=CHECK-ROWS
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/index.ll 2>&1 | FileCheck %s --check-prefix=CHECK-INDEX
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/constant.ll 2>&1 | FileCheck %s --check-prefix=CHECK-CONSTANT
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/undefined.ll 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/allocated.ll 2>&1 | FileCheck %s --check-prefix=CHECK-ALLOCATED
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/duplicate-index.ll 2>&1 | FileCheck %s --check-prefix=CHECK-DUPLICATE-INDEX
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/id.ll 2>&1 | FileCheck %s --check-prefix=CHECK-ID
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/vertex.ll 2>&1 | FileCheck %s --check-prefix=CHECK-VERTEX
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/output.ll 2>&1 | FileCheck %s --check-prefix=CHECK-OUTPUT
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/sm60.ll 2>&1 | FileCheck %s --check-prefix=CHECK-SM60
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/access-type.ll 2>&1 | FileCheck %s --check-prefix=CHECK-ACCESS-TYPE
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/access-col.ll 2>&1 | FileCheck %s --check-prefix=CHECK-ACCESS-COL
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/access-row.ll 2>&1 | FileCheck %s --check-prefix=CHECK-ACCESS-ROW

; Each case is a complete module with one invalid SV_Barycentrics declaration.
; Unallocated elements must undergo the same semantic validation as packed
; elements.

;--- integer.ll
; CHECK-INTEGER: SV_Barycentrics requires one row of three 16- or 32-bit floating-point components
; Integer component type.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 5, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- cols.ll
; CHECK-COLS: SV_Barycentrics requires one row of three 16- or 32-bit floating-point components
; Two columns instead of three.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 2, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- rows.ll
; CHECK-ROWS: SV_Barycentrics requires one row of three 16- or 32-bit floating-point components
; Two rows instead of one.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 2, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0, i32 1}

;--- index.ll
; CHECK-INDEX: SV_Barycentrics requires semantic index 0
; Semantic index 1.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 1}

;--- constant.ll
; CHECK-CONSTANT: SV_Barycentrics requires perspective or noperspective interpolation
; Constant interpolation.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 1, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- undefined.ll
; CHECK-UNDEFINED: SV_Barycentrics requires perspective or noperspective interpolation
; Undefined interpolation.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 0, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- allocated.ll
; CHECK-ALLOCATED: SV_Barycentrics must not have an allocated location
; Allocated at row 0, column 0.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- duplicate-index.ll
; CHECK-DUPLICATE-INDEX: duplicate semantic name and index
; Index 0 is declared twice, with differently cased names.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2, !4}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}
!4 = !{i32 1, !"sv_barycentrics", i32 9, i32 28, !3, i32 4, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}

;--- id.ll
; CHECK-ID: signature IDs must be dense and in list order
; An unallocated element still occupies ID 0, so the next ID must be 1.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2, !4}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}
!4 = !{i32 2, !"A", i32 9, i32 0, !3, i32 2, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}

;--- vertex.ll
; CHECK-VERTEX: invalid or unsupported semantic for this stage and signature category
; Vertex shader input.
target triple = "dxil-pc-shadermodel6.1-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- output.ll
; CHECK-OUTPUT: invalid or unsupported semantic for this stage and signature category
; Pixel shader output.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- sm60.ll
; CHECK-SM60: SV_Barycentrics requires shader model 6.1
; Shader model 6.0.
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- access-type.ll
; CHECK-ACCESS-TYPE: input signature element 0 ('SV_Barycentrics'): signature access type disagrees with its element
; A half load from a float element.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 {
  %x = call half @llvm.dx.load.input.f16(i32 0, i32 0, i8 0, i32 poison)
  ret void
}
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- access-col.ll
; CHECK-ACCESS-COL: input signature element 0 ('SV_Barycentrics'): signature access has an invalid component index
; Component 3 of a three-component element.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 3, i32 poison)
  ret void
}
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- access-row.ll
; CHECK-ACCESS-ROW: input signature element 0 ('SV_Barycentrics'): signature access has an invalid row index
; Row 1 of a single-row element.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 0, i32 1, i8 0, i32 poison)
  ret void
}
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}
