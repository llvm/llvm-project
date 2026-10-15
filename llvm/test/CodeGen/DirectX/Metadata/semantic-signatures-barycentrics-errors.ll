; RUN: split-file %s %t
; RUN: cat %t/common.ll %t/integer.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=TYPE
; RUN: cat %t/common.ll %t/cols.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=TYPE
; RUN: cat %t/common.ll %t/rows.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=TYPE
; RUN: cat %t/common.ll %t/index.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=INDEX
; RUN: cat %t/common.ll %t/constant.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=INTERPOLATION
; RUN: cat %t/common.ll %t/undefined.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=INTERPOLATION
; RUN: cat %t/common.ll %t/allocated.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=ALLOCATED
; RUN: cat %t/common.ll %t/duplicate-index.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=DUPLICATE
; RUN: cat %t/common.ll %t/id.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=ID
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/vertex.ll 2>&1 | FileCheck %s --check-prefix=STAGE
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/output.ll 2>&1 | FileCheck %s --check-prefix=STAGE
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/sm60.ll 2>&1 | FileCheck %s --check-prefix=SM
; RUN: cat %t/access-common.ll %t/access-type.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=ACCESS-TYPE
; RUN: cat %t/access-common.ll %t/access-col.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=ACCESS-COL
; RUN: cat %t/access-common.ll %t/access-row.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=ACCESS-ROW

; TYPE: SV_Barycentrics requires one row of three 16- or 32-bit floating-point components
; INDEX: SV_Barycentrics requires semantic index 0
; INTERPOLATION: SV_Barycentrics requires perspective or noperspective interpolation
; ALLOCATED: SV_Barycentrics must not have an allocated location
; DUPLICATE: duplicate semantic name and index
; ID: signature IDs must be dense and in list order
; STAGE: invalid or unsupported semantic for this stage and signature category
; SM: SV_Barycentrics requires shader model 6.1
; ACCESS-TYPE: input signature element 0 ('SV_Barycentrics'): signature access type disagrees with its element
; ACCESS-COL: input signature element 0 ('SV_Barycentrics'): signature access has an invalid component index
; ACCESS-ROW: input signature element 0 ('SV_Barycentrics'): signature access has an invalid row index

;--- common.ll
; Individual cases supply the element list so malformed unallocated elements
; must undergo the same semantic validation as packed elements.
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!3 = !{i32 0}

;--- integer.ll
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 5, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}

;--- cols.ll
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 2, i32 -1, i8 -1, i8 0, i8 0, i32 0}

;--- rows.ll
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !5, i32 2, i32 2, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!5 = !{i32 0, i32 1}

;--- index.ll
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !5, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!5 = !{i32 1}

;--- constant.ll
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 1, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}

;--- undefined.ll
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 0, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}

;--- allocated.ll
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 0, i8 0, i8 0, i8 0, i32 0}

;--- duplicate-index.ll
!1 = !{!2, !5}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!5 = !{i32 1, !"sv_barycentrics", i32 9, i32 28, !3, i32 4, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}

;--- id.ll
; An unallocated element still occupies ID 0, so the next ID must be 1.
!1 = !{!2, !5}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!5 = !{i32 2, !"A", i32 9, i32 0, !3, i32 2, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}

;--- vertex.ll
target triple = "dxil-pc-shadermodel6.1-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- output.ll
target triple = "dxil-pc-shadermodel6.1-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- sm60.ll
target triple = "dxil-pc-shadermodel6.0-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- access-common.ll
target triple = "dxil-pc-shadermodel6.1-pixel"
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- access-type.ll
define void @main() #0 {
  %x = call half @llvm.dx.load.input.f16(i32 0, i32 0, i8 0, i32 poison)
  ret void
}

;--- access-col.ll
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 3, i32 poison)
  ret void
}

;--- access-row.ll
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 0, i32 1, i8 0, i32 poison)
  ret void
}
