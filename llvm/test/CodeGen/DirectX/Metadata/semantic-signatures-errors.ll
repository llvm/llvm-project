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
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/wrong-kind.ll 2>&1 | FileCheck %s --check-prefix=KIND
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/integer-undefined.ll 2>&1 | FileCheck %s --check-prefix=INTEGER-UNDEFINED
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/integer-linear.ll 2>&1 | FileCheck %s --check-prefix=INTEGER-LINEAR
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/vertex-location.ll 2>&1 | FileCheck %s --check-prefix=VERTEX
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/target-location.ll 2>&1 | FileCheck %s --check-prefix=TARGET
; RUN: cat %t/access-common.ll %t/access-id.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=ACCESS-ID
; RUN: cat %t/access-common.ll %t/access-dynamic-id.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=ACCESS-DYNAMIC-ID
; RUN: cat %t/access-common.ll %t/access-component.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=ACCESS-COMPONENT
; RUN: cat %t/access-common.ll %t/access-row.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=ACCESS-ROW
; RUN: cat %t/access-common.ll %t/access-dynamic-row.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=ACCESS-DYNAMIC-ROW
; RUN: cat %t/access-common.ll %t/access-type.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=ACCESS-TYPE
; RUN: cat %t/access-common.ll %t/access-output.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=ACCESS-OUTPUT

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
!2 = !{i32 0, !"A", i32 9, i32 0, !3, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
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
; OVERFLOW: entry 'main' output signature: allocated signature element exceeds register bounds
; Invalid start row (operand 8): B starts at 32, beyond the register space.
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2, !4}
!2 = !{i32 0, !"A", i32 9, i32 0, !3, i32 0, i32 32, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31}
!4 = !{i32 1, !"B", i32 9, i32 0, !5, i32 0, i32 1, i8 1, i32 32, i8 0, i8 0, i8 0, i32 0}
!5 = !{i32 0}

;--- partial.ll
; PARTIAL: entry 'main' input signature: signature element 1 ('B') has no allocated location
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

;--- wrong-kind.ll
; KIND: entry 'main' input signature: semantic name and kind disagree at this signature point
; Invalid kind (operand 3): Position (3) instead of Arbitrary (0) at a VS input.
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Position", i32 9, i32 3, !3, i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- integer-undefined.ll
; INTEGER-UNDEFINED: entry 'main' input signature: integer pixel inputs require constant interpolation
; Invalid mode (operand 5): Undefined (0) must not be repaired to Constant (1).
target triple = "dxil-pc-shadermodel6.8-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"I", i32 4, i32 0, !3, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- integer-linear.ll
; INTEGER-LINEAR: entry 'main' input signature: integer pixel inputs require constant interpolation
; Invalid mode (operand 5): Linear (2) for an integer pixel input.
target triple = "dxil-pc-shadermodel6.8-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"I", i32 4, i32 0, !3, i32 2, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- vertex-location.ll
; Invalid row (operand 8): the first vertex input must start at row 0.
; Report the invalid layout rather than relocating it.
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
; VERTEX: entry 'main' input signature: vertex inputs require stacked locations in signature order (element 0)
!2 = !{i32 0, !"A", i32 9, i32 0, !3, i32 0, i32 1, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- target-location.ll
; Invalid row (operand 8): SV_Target3 must use row 3, not row 1.
target triple = "dxil-pc-shadermodel6.8-pixel"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2}
; TARGET: entry 'main' output signature: pixel target location must match its semantic index at column 0
!2 = !{i32 0, !"SV_Target", i32 9, i32 16, !3, i32 0, i32 1, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 3}

;--- access-common.ll
target triple = "dxil-pc-shadermodel6.8-vertex"
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !1}
!1 = !{!2, !3}
!2 = !{i32 0, !"A", i32 9, i32 0, !4, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 1, !"B", i32 9, i32 0, !5, i32 0, i32 2, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
!4 = !{i32 0}
!5 = !{i32 0, i32 1}

;--- access-id.ll
; ACCESS-ID: entry 'main': input signature: signature access has an invalid element ID 123456789
; The intrinsic ID is outside the signature's two-element list.
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 123456789, i32 0, i8 0, i32 poison)
  ret void
}

;--- access-dynamic-id.ll
; ACCESS-DYNAMIC-ID: entry 'main': input signature: signature access requires a constant element ID
; Signature IDs cannot be selected dynamically.
define void @main(i32 %id) #0 {
  %x = call float @llvm.dx.load.input.f32(i32 %id, i32 0, i8 0, i32 poison)
  ret void
}

;--- access-component.ll
; ACCESS-COMPONENT: entry 'main': input signature element 1 ('B'): signature access has an invalid component index
; Component 123 is outside B's one-column extent.
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 1, i32 0, i8 123, i32 poison)
  ret void
}

;--- access-row.ll
; ACCESS-ROW: entry 'main': input signature element 1 ('B'): signature access has an invalid row index
; Row 123456789 is outside B's two-row extent.
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 1, i32 123456789, i8 0, i32 poison)
  ret void
}

;--- access-dynamic-row.ll
; ACCESS-DYNAMIC-ROW: entry 'main': input signature element 0 ('A'): dynamic indexing requires a multi-row signature element
; A has only one row, so its row cannot be selected dynamically.
define void @main(i32 %row) #0 {
  %x = call float @llvm.dx.load.input.f32(i32 0, i32 %row, i8 0, i32 poison)
  ret void
}

;--- access-type.ll
; ACCESS-TYPE: entry 'main': input signature element 1 ('B'): signature access type disagrees with its element
; B has F32 components, not i32 components.
define void @main() #0 {
  %x = call i32 @llvm.dx.load.input.i32(i32 1, i32 0, i8 0, i32 poison)
  ret void
}

;--- access-output.ll
; ACCESS-OUTPUT: entry 'main': output signature element 1 ('B'): signature access has an invalid component index
; Output diagnostics must distinguish the output signature from the input.
define void @main() #0 {
  call void @llvm.dx.store.output.f32(i32 1, i32 0, i8 123, float 0.0)
  ret void
}
