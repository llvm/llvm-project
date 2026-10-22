; RUN: split-file %s %t
; RUN: llc -filetype=obj %t/v16.ll -o %t/16.dxbc
; RUN: obj2yaml %t/16.dxbc | FileCheck %s --check-prefixes=CHECK,CHECK-UNPADDED
; RUN: llc -filetype=obj %t/v17.ll -o - | obj2yaml | FileCheck %s --check-prefixes=CHECK,CHECK-PADDED

; Round trip the unpadded container through yaml2obj to check that the
; explicit (unpadded) part sizes are preserved and that the following,
; unaligned, PSV0 part is still readable.
; RUN: obj2yaml %t/16.dxbc | yaml2obj | obj2yaml | FileCheck %s --check-prefixes=CHECK,CHECK-UNPADDED

; Each case is a complete module that differs only in its validator version.
; Validator 1.7 pads signature parts to a four-byte boundary; older versions
; do not. The differing name lengths leave PSV0 at an unaligned offset when
; unpadded.
; CHECK:               Name: ISG1
; CHECK-UNPADDED-NEXT: Size: 42
; CHECK-PADDED-NEXT:   Size: 44
; CHECK: Name: A
; CHECK:               Name: OSG1
; CHECK-UNPADDED-NEXT: Size: 43
; CHECK-PADDED-NEXT:   Size: 44
; CHECK: Name: BB
; CHECK: Name: PSV0
; CHECK:      SigInputElements:
; CHECK-NEXT: - Name: A
; CHECK:      SigOutputElements:
; CHECK-NEXT: - Name: BB

;--- v16.ll
target triple = "dxil-pc-shadermodel6.0-vertex"
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %x)
  ret void
}
attributes #0 = { "hlsl.shader"="vertex" }
!dx.valver = !{!0}
!dx.semantic.signatures = !{!1}
!0 = !{i32 1, i32 6}
!1 = !{ptr @main, !2, !3}
!2 = !{!4}
!3 = !{!5}
!4 = !{i32 0, !"A", i32 9, i32 0, !6, i32 0, i32 1, i8 1, i32 0, i8 0, i8 15, i8 15, i32 0}
!5 = !{i32 0, !"BB", i32 9, i32 0, !6, i32 0, i32 1, i8 1, i32 0, i8 0, i8 15, i8 15, i32 0}
!6 = !{i32 0}

;--- v17.ll
target triple = "dxil-pc-shadermodel6.0-vertex"
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %x)
  ret void
}
attributes #0 = { "hlsl.shader"="vertex" }
!dx.valver = !{!0}
!dx.semantic.signatures = !{!1}
!0 = !{i32 1, i32 7}
!1 = !{ptr @main, !2, !3}
!2 = !{!4}
!3 = !{!5}
!4 = !{i32 0, !"A", i32 9, i32 0, !6, i32 0, i32 1, i8 1, i32 0, i8 0, i8 15, i8 15, i32 0}
!5 = !{i32 0, !"BB", i32 9, i32 0, !6, i32 0, i32 1, i8 1, i32 0, i8 0, i8 15, i8 15, i32 0}
!6 = !{i32 0}
