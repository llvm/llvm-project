; RUN: split-file %s %t
; RUN: cat %t/shader.ll %t/v10.ll > %t/10.ll
; RUN: cat %t/shader.ll %t/v14.ll > %t/14.ll
; RUN: cat %t/shader.ll %t/v16.ll > %t/16.ll
; RUN: cat %t/shader.ll %t/v17.ll > %t/17.ll
; RUN: llc -filetype=obj %t/10.ll -o - | obj2yaml | FileCheck %s --check-prefixes=PARTS,LEGACY,PSV0
; RUN: llc -filetype=obj %t/14.ll -o - | obj2yaml | FileCheck %s --check-prefixes=PARTS,LEGACY,PSV1
; RUN: llc -filetype=obj %t/16.ll -o - | obj2yaml | FileCheck %s --check-prefixes=PARTS,MODERN,PSV2
; RUN: llc -filetype=obj %t/17.ll -o - | obj2yaml | FileCheck %s --check-prefixes=PARTS,MODERN,PSV2

; Check validator-version-dependent signature masks and PSV records.
; Metadata translation and signature padding are tested separately in
; SemanticSignatures-metadata-versions.ll and SemanticSignatures-padding.ll.

; PARTS: Name: ISG1
; LEGACY: ExclusiveMask: 0
; MODERN: ExclusiveMask: 1
; PARTS: Name: OSG1
; LEGACY: ExclusiveMask: 254
; MODERN: ExclusiveMask: 14
; PSV0: Name: PSV0
; PSV0: Version: 0
; PSV0-NOT: SigInputVectors:
; PSV0-NOT: SigOutputVectors:
; PSV0-NOT: SigInputElements:
; PSV0-NOT: SigOutputElements:
; PSV0-NOT: InputOutputMap:
; PSV1: Name: PSV0
; PSV1: Version: 1
; PSV1: SigInputVectors: 1
; PSV1-NEXT: SigOutputVectors: [ 1, 0, 0, 0 ]
; PSV2: Name: PSV0
; PSV2: Version: 2
; PSV2: SigInputVectors: 1
; PSV2-NEXT: SigOutputVectors: [ 1, 0, 0, 0 ]

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
; Signatures have frontend-provided locations. Stale masks are ignored.
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
