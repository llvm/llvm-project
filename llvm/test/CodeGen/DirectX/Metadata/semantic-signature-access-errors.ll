; RUN: split-file %s %t
; RUN: cat %t/common.ll %t/id.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=ID
; RUN: cat %t/common.ll %t/dynamic-id.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=DYNAMIC-ID
; RUN: cat %t/common.ll %t/component.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=COMPONENT
; RUN: cat %t/common.ll %t/row.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=ROW
; RUN: cat %t/common.ll %t/dynamic-row.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=DYNAMIC-ROW
; RUN: cat %t/common.ll %t/type.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=TYPE
; RUN: cat %t/common.ll %t/output.ll > %t/test.ll
; RUN: not opt -disable-output -passes=dxil-translate-metadata %t/test.ll 2>&1 | FileCheck %s --check-prefix=OUTPUT

;--- common.ll
target triple = "dxil-pc-shadermodel6.8-vertex"
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !1}
!1 = !{!2, !3}
!2 = !{i32 0, !"A", i32 9, i32 0, !4, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 1, !"B", i32 9, i32 0, !5, i32 0, i32 2, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!4 = !{i32 0}
!5 = !{i32 0, i32 1}

;--- id.ll
; ID: entry 'main': input signature: signature access has an invalid element ID 123456789
; The intrinsic ID is outside the signature's two-element list.
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 123456789, i32 0, i8 0, i32 poison)
  ret void
}

;--- dynamic-id.ll
; DYNAMIC-ID: entry 'main': input signature: signature access requires a constant element ID
; Signature IDs cannot be selected dynamically.
define void @main(i32 %id) #0 {
  %x = call float @llvm.dx.load.input.f32(i32 %id, i32 0, i8 0, i32 poison)
  ret void
}

;--- component.ll
; COMPONENT: entry 'main': input signature element 1 ('B'): signature access has an invalid component index
; Component 123 is outside B's one-column extent.
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 1, i32 0, i8 123, i32 poison)
  ret void
}

;--- row.ll
; ROW: entry 'main': input signature element 1 ('B'): signature access has an invalid row index
; Row 123456789 is outside B's two-row extent.
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 1, i32 123456789, i8 0, i32 poison)
  ret void
}

;--- dynamic-row.ll
; DYNAMIC-ROW: entry 'main': input signature element 0 ('A'): dynamic indexing requires a multi-row signature element
; A has only one row, so its row cannot be selected dynamically.
define void @main(i32 %row) #0 {
  %x = call float @llvm.dx.load.input.f32(i32 0, i32 %row, i8 0, i32 poison)
  ret void
}

;--- type.ll
; TYPE: entry 'main': input signature element 1 ('B'): signature access type disagrees with its element
; B has F32 components, not i32 components.
define void @main() #0 {
  %x = call i32 @llvm.dx.load.input.i32(i32 1, i32 0, i8 0, i32 poison)
  ret void
}

;--- output.ll
; OUTPUT: entry 'main': output signature element 1 ('B'): signature access has an invalid component index
; Output diagnostics must distinguish the output signature from the input.
define void @main() #0 {
  call void @llvm.dx.store.output.f32(i32 1, i32 0, i8 123, float 0.0)
  ret void
}
