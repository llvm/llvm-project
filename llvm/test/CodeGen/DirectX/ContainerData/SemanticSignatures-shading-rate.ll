; RUN: split-file %s %t
; RUN: llc -O0 -filetype=obj %t/vs.ll -o %t-vs.dxbc
; RUN: obj2yaml %t-vs.dxbc | FileCheck %s --check-prefix=VS
; Container must survive obj2yaml -> yaml2obj -> obj2yaml round trip.
; RUN: obj2yaml %t-vs.dxbc | yaml2obj | obj2yaml | FileCheck %s --check-prefix=VS
; RUN: llc -O0 -filetype=obj %t/ps.ll -o %t-ps.dxbc
; RUN: obj2yaml %t-ps.dxbc | FileCheck %s --check-prefix=PS
; RUN: obj2yaml %t-ps.dxbc | yaml2obj | obj2yaml | FileCheck %s --check-prefix=PS

; SV_ShadingRate is a vertex output and a pixel input (SM 6.4+). The metadata
; keeps the source spelling, but the container must use the canonical name.

;--- vs.ll
target triple = "dxil-pc-shadermodel6.8-vertex"

define void @main() #0 {
  %p = call <4 x float> @llvm.dx.load.input.v4f32(i32 0, i32 0, i8 0, i32 poison)
  call void @llvm.dx.store.output.v4f32(i32 0, i32 0, i8 0, <4 x float> %p)
  call void @llvm.dx.store.output.i32(i32 1, i32 0, i8 0, i32 5)
  ret void
}

attributes #0 = { "hlsl.shader"="vertex" }

!dx.valver = !{!20}
!20 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3}
!2 = !{!4, !5}
!3 = !{i32 0, !"POSITION", i32 9, i32 0, !10, i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
!4 = !{i32 0, !"SV_POSITION", i32 9, i32 3, !10, i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
!5 = !{i32 1, !"SV_SHADINGRATE", i32 5, i32 29, !10, i32 0, i32 1, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
!10 = !{i32 0}

; VS: Name: OSG1
; VS: Name: SV_Position
; VS: SystemValue: Position
; VS: Name: SV_ShadingRate
; VS: SystemValue: ShadingRate
; VS-NEXT: CompType: UInt32
; VS-NEXT: Register: 1
; VS-NEXT: Mask: 1
; VS: Name: PSV0
; VS: SigOutputElements:
; VS: Kind: ShadingRate

;--- ps.ll
target triple = "dxil-pc-shadermodel6.8-pixel"

define void @main() #0 {
  %r = call i32 @llvm.dx.load.input.i32(i32 1, i32 0, i8 0, i32 poison)
  %f = uitofp i32 %r to float
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %f)
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

!dx.valver = !{!20}
!20 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3, !4}
!2 = !{!5}
!3 = !{i32 0, !"SV_POSITION", i32 9, i32 3, !10, i32 4, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
; Integer pixel inputs require constant interpolation.
!4 = !{i32 1, !"SV_SHADINGRATE", i32 5, i32 29, !10, i32 1, i32 1, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
!5 = !{i32 0, !"SV_TARGET", i32 9, i32 16, !11, i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
!10 = !{i32 0}
!11 = !{i32 0}

; PS: Name: ISG1
; PS: Name: SV_Position
; PS: Name: SV_ShadingRate
; PS: SystemValue: ShadingRate
; PS-NEXT: CompType: UInt32
; PS: Name: PSV0
; PS: SigInputElements:
; PS: Kind: ShadingRate
