; RUN: split-file %s %t
; RUN: cat %t/shader.ll %t/v10.ll > %t/10.ll
; RUN: cat %t/shader.ll %t/v14.ll > %t/14.ll
; RUN: cat %t/shader.ll %t/v17.ll > %t/17.ll
; RUN: opt -S -dxil-translate-metadata %t/10.ll | FileCheck %s --check-prefix=MD10
; RUN: opt -S -dxil-translate-metadata %t/14.ll | FileCheck %s --check-prefix=MD14
; RUN: llc -filetype=obj %t/10.ll -o - | obj2yaml | FileCheck %s --check-prefixes=LEGACY,PSV0
; RUN: llc -filetype=obj %t/14.ll -o - | obj2yaml | FileCheck %s --check-prefixes=LEGACY,PSV1
; RUN: llc -filetype=obj %t/17.ll -o - | obj2yaml | FileCheck %s --check-prefixes=MODERN,PSV2

; MD10-NOT: !dx.viewIdState
; MD10-NOT: !dx.semantic.signatures
; MD10: !dx.entryPoints =
; MD10-NOT: !dx.viewIdState
; MD10-NOT: !dx.semantic.signatures
; MD14: !dx.viewIdState =
; MD14: !{i32 0, !"A", i8 9, i8 0, !{{[0-9]+}}, i8 0, i32 1, i8 1, i32 0, i8 0, null}
; MD14: !{i32 0, !"B", i8 9, i8 0, !{{[0-9]+}}, i8 0, i32 1, i8 1, i32 0, i8 0, null}

; LEGACY: Name: ISG1
; LEGACY: ExclusiveMask: 0
; LEGACY: Name: OSG1
; LEGACY: ExclusiveMask: 254
; MODERN: Name: ISG1
; MODERN: ExclusiveMask: 1
; MODERN: Name: OSG1
; MODERN: ExclusiveMask: 14
; PSV0: Name: PSV0
; PSV0: Version: 0
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
; Already allocated canonical signatures are accepted. Stale masks are ignored.
!3 = !{i32 0, !"A", i32 9, i32 0, !5, i32 0, i32 1, i8 1, i32 0, i8 0, i8 15, i8 15, i32 0}
!4 = !{i32 0, !"B", i32 9, i32 0, !5, i32 0, i32 1, i8 1, i32 0, i8 0, i8 15, i8 15, i32 0}
!5 = !{i32 0}

;--- v10.ll
!dx.valver = !{!99}
!99 = !{i32 1, i32 0}
;--- v14.ll
!dx.valver = !{!99}
!99 = !{i32 1, i32 4}
;--- v17.ll
!dx.valver = !{!99}
!99 = !{i32 1, i32 7}
