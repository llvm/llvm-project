; RUN: split-file %s %t
; RUN: cat %t/shader.ll %t/v16.ll > %t/16.ll
; RUN: cat %t/shader.ll %t/v17.ll > %t/17.ll
; RUN: llc -filetype=obj %t/16.ll -o %t/16.dxbc
; RUN: obj2yaml %t/16.dxbc | FileCheck %s --check-prefixes=CHECK,UNPADDED
; RUN: llc -filetype=obj %t/17.ll -o - | obj2yaml | FileCheck %s --check-prefixes=CHECK,PADDED
; RUN: obj2yaml %t/16.dxbc | yaml2obj | obj2yaml | FileCheck %s --check-prefixes=CHECK,UNPADDED

; Validator 1.7 pads signature parts to a four-byte boundary but older versions
; do not. The differing name lengths leave PSV0 at an unaligned offset when
; unpadded.
; CHECK: Name: ISG1
; UNPADDED-NEXT: Size: 42
; PADDED-NEXT: Size: 44
; CHECK: Name: A
; CHECK: Name: OSG1
; UNPADDED-NEXT: Size: 43
; PADDED-NEXT: Size: 44
; CHECK: Name: BB
; CHECK: Name: PSV0
; CHECK: SigInputElements:
; CHECK-NEXT: - Name: A
; CHECK: SigOutputElements:
; CHECK-NEXT: - Name: BB

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
!4 = !{i32 0, !"BB", i32 9, i32 0, !5, i32 0, i32 1, i8 1, i32 0, i8 0, i8 15, i8 15, i32 0}
!5 = !{i32 0}

;--- v16.ll
!dx.valver = !{!99}
!99 = !{i32 1, i32 6}
;--- v17.ll
!dx.valver = !{!99}
!99 = !{i32 1, i32 7}
