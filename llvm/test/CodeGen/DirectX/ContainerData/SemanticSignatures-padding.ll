; RUN: split-file %s %t
; RUN: cat %t/shader.ll %t/v10.ll > %t/10.ll
; RUN: cat %t/shader.ll %t/v14.ll > %t/14.ll
; RUN: cat %t/shader.ll %t/v16.ll > %t/16.ll
; RUN: cat %t/shader.ll %t/v17.ll > %t/17.ll
; RUN: llc -filetype=obj %t/10.ll -o - | obj2yaml | FileCheck %s --check-prefixes=PARTS,UNPADDED
; RUN: llc -filetype=obj %t/14.ll -o - | obj2yaml | FileCheck %s --check-prefixes=PARTS,UNPADDED
; RUN: llc -filetype=obj %t/16.ll -o - | obj2yaml | FileCheck %s --check-prefixes=PARTS,UNPADDED
; RUN: llc -filetype=obj %t/17.ll -o - | obj2yaml | FileCheck %s --check-prefixes=PARTS,PADDED
; RUN: cat %t/output.ll %t/v14.ll > %t/output-14.ll
; RUN: cat %t/output.ll %t/v16.ll > %t/output-16.ll
; RUN: cat %t/output.ll %t/v17.ll > %t/output-17.ll
; RUN: llc -filetype=obj %t/output-14.ll -o - | obj2yaml | FileCheck %s --check-prefixes=OUTPUT,OUTPUT-UNPADDED
; RUN: llc -filetype=obj %t/output-16.ll -o %t.output.dxbc
; RUN: obj2yaml %t.output.dxbc | FileCheck %s --check-prefixes=OUTPUT,OUTPUT-UNPADDED
; RUN: obj2yaml %t.output.dxbc | yaml2obj | obj2yaml | FileCheck %s --check-prefixes=OUTPUT,OUTPUT-UNPADDED
; RUN: llc -filetype=obj %t/output-17.ll -o - | obj2yaml | FileCheck %s --check-prefixes=OUTPUT,OUTPUT-PADDED

; Validator 1.7 pads signature parts to a four-byte boundary; older versions do not.
; PARTS: Name: ISG1
; UNPADDED-NEXT: Size: 42
; PADDED-NEXT: Size: 44
; PARTS: Name: OSG1
; UNPADDED-NEXT: Size: 42
; PADDED-NEXT: Size: 44

; A single unpadded signature leaves the following PSV part unaligned.
; Check that the PSV output signature survives emission and YAML round-tripping.
; OUTPUT: Name: ISG1
; OUTPUT-NEXT: Size: 8
; OUTPUT: Name: OSG1
; OUTPUT-UNPADDED-NEXT: Size: 42
; OUTPUT-PADDED-NEXT: Size: 44
; OUTPUT: Name: A
; OUTPUT: Name: PSV0
; OUTPUT: SigOutputVectors: [ 1, 0, 0, 0 ]
; OUTPUT: SigOutputElements:
; OUTPUT-NEXT: - Name: A

;--- output.ll
target triple = "dxil-pc-shadermodel6.0-vertex"
define void @main() #0 {
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float 1.0)
  ret void
}
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2}
!2 = !{i32 0, !"A", i32 9, i32 0, !3, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- shader.ll
target triple = "dxil-pc-shadermodel6.0-vertex"
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %x)
  ret void
}
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{i32 0, !"A", i32 9, i32 0, !5, i32 0, i32 1, i8 1, i32 0, i8 0, i8 15, i8 15, i32 0}
!4 = !{i32 0, !"B", i32 9, i32 0, !5, i32 0, i32 1, i8 1, i32 0, i8 0, i8 15, i8 15, i32 0}
!5 = !{i32 0}

;--- v10.ll
!dx.valver = !{!99}
!99 = !{i32 1, i32 0}
;--- v14.ll
!dx.valver = !{!99}
!99 = !{i32 1, i32 4}
;--- v16.ll
!dx.valver = !{!99}
!99 = !{i32 1, i32 6}
;--- v17.ll
!dx.valver = !{!99}
!99 = !{i32 1, i32 7}
