; RUN: split-file %s %t
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/triple.ll 2>&1 | FileCheck %s --check-prefix=TRIPLE
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/duplicate.ll 2>&1 | FileCheck %s --check-prefix=DUPLICATE
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/width.ll 2>&1 | FileCheck %s --check-prefix=WIDTH
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/id.ll 2>&1 | FileCheck %s --check-prefix=ID
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/rows.ll 2>&1 | FileCheck %s --check-prefix=ROWS
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/component.ll 2>&1 | FileCheck %s --check-prefix=COMPONENT
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/stage.ll 2>&1 | FileCheck %s --check-prefix=STAGE
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/access.ll 2>&1 | FileCheck %s --check-prefix=ACCESS
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/missing.ll 2>&1 | FileCheck %s --check-prefix=MISSING
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/overflow.ll 2>&1 | FileCheck %s --check-prefix=OVERFLOW
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/partial.ll 2>&1 | FileCheck %s --check-prefix=PARTIAL
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/name.ll 2>&1 | FileCheck %s --check-prefix=NAME
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/function.ll 2>&1 | FileCheck %s --check-prefix=FUNCTION

;--- triple.ll
; TRIPLE: Invalid semantic signature: expected an entry/input/output signature triple
; Invalid record shape: four operands instead of entry/input/output.
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, null, null}

;--- duplicate.ll
; DUPLICATE: Invalid semantic signature: duplicate signature record for entry 'main'
; Invalid table: the same entry record appears twice.
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0, !0}
!0 = !{ptr @main, null, null}

;--- width.ll
; WIDTH: entry 'main' input signature: expected i32 at operand 0
; Invalid signature ID width (operand 0): i128 instead of i32.
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i128 18446744073709551616, !"A", i32 9, i32 0, !3, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- id.ll
; ID: entry 'main' input signature: signature IDs must be dense and in list order
; Invalid signature ID (operand 0): the first element must have ID 0.
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 123456789, !"A", i32 9, i32 0, !3, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- rows.ll
; ROWS: entry 'main' input signature: signature row count must be within 1-32
; Invalid row count (operand 6): zero rows, with a matching empty index list.
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"A", i32 9, i32 0, !3, i32 0, i32 0, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{}

;--- component.ll
; COMPONENT: entry 'main' input signature: unsupported signature component type
; Unsupported component type (operand 2): F64 (10), rather than F32 (9).
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"A", i32 10, i32 0, !3, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- stage.ll
; STAGE: entry 'main' input signature: nonempty signatures are currently supported only for vertex and pixel entries
; Unsupported entry stage: geometry rather than vertex or pixel.
target triple = "dxil-pc-shadermodel6.8-geometry"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="geometry" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"A", i32 9, i32 0, !3, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- access.ll
; ACCESS: entry 'main': input signature element 0 ('A'): signature access has an invalid component index
; Invalid access column (intrinsic operand 2): outside this scalar element.
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 123, i32 poison)
  ret void
}
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"A", i32 9, i32 0, !3, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- missing.ll
; MISSING: signature access in function 'main' has no entry signature metadata
; Missing signature table for an entry that accesses an input.
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  ret void
}
attributes #0 = { "hlsl.shader"="vertex" }

;--- overflow.ll
; OVERFLOW: entry 'main' output signature: signature elements do not fit in 32 rows (element 1)
; Invalid total extent: A fills all 32 rows, leaving no space for B.
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2, !4}
!2 = !{i32 0, !"A", i32 9, i32 0, !3, i32 0, i32 32, i8 4, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31}
!4 = !{i32 1, !"B", i32 9, i32 0, !5, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!5 = !{i32 0}

;--- partial.ll
; PARTIAL: entry 'main' input signature: partially allocated signatures are not supported
; Invalid mixed allocation: A has a location (operands 8/9), but B does not.
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2, !4}
!2 = !{i32 0, !"A", i32 9, i32 0, !3, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}
!4 = !{i32 1, !"B", i32 9, i32 0, !3, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}

;--- name.ll
; NAME: entry 'main' input signature: expected semantic name string
; Invalid semantic name (operand 1): null instead of a metadata string.
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, null, i32 9, i32 0, !3, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- function.ll
; FUNCTION: signature entry must be a defined function in the module
; Invalid entry (record operand 0): a global variable instead of a function.
target triple = "dxil-pc-shadermodel6.8-vertex"
@global = global i32 0
!dx.semantic.signatures = !{!0}
!0 = !{ptr @global, null, null}
