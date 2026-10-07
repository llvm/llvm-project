; RUN: llc -O0 -filetype=obj %s -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %s --check-prefix=PARTS
; RUN: obj2yaml %t.dxbc | yaml2obj | obj2yaml | FileCheck %s --check-prefix=PARTS
; RUN: llc -O2 -filetype=obj %s -o - | obj2yaml | FileCheck %s --check-prefix=PARTS

target triple = "dxil-pc-shadermodel6.8-vertex"

define void @main() #0 {
  %p = call <4 x float> @llvm.dx.load.input.v4f32(i32 0, i32 0, i8 0, i32 poison)
  %uv = call <2 x float> @llvm.dx.load.input.v2f32(i32 1, i32 0, i8 0, i32 poison)
  %u = extractelement <2 x float> %uv, i32 0
  call void @llvm.dx.store.output.v4f32(i32 0, i32 0, i8 0, <4 x float> %p)
  call void @llvm.dx.store.output.f32(i32 1, i32 0, i8 0, float %u)
  call void @llvm.dx.store.output.v2f32(i32 2, i32 0, i8 0, <2 x float> %uv)
  ret void
}

attributes #0 = { "hlsl.shader"="vertex" }

!dx.valver = !{!20}
!20 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3, !4}
!2 = !{!5, !6, !7}
!3 = !{i32 0, !"POSITION", i32 9, i32 0, !10, i32 0, i32 1, i8 4, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!4 = !{i32 1, !"TEXCOORD", i32 9, i32 0, !10, i32 0, i32 1, i8 2, i32 -1, i8 -1, i8 0, i8 0, i32 0}
; The metadata keeps the source spelling, but OSG1 must use SV_Position.
!5 = !{i32 0, !"SV_POSITION", i32 9, i32 3, !10, i32 0, i32 1, i8 4, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!6 = !{i32 1, !"A", i32 9, i32 0, !10, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!7 = !{i32 2, !"B", i32 9, i32 0, !10, i32 0, i32 1, i8 2, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!10 = !{i32 0}

; PARTS: Name: ISG1
; PARTS: Name: POSITION
; PARTS: SystemValue: Undefined
; PARTS-NEXT: CompType: Float32
; PARTS-NEXT: Register: 0
; PARTS-NEXT: Mask: 15
; PARTS-NEXT: ExclusiveMask: 15
; PARTS: Name: TEXCOORD
; PARTS: Register: 1
; PARTS-NEXT: Mask: 3
; PARTS-NEXT: ExclusiveMask: 3
; PARTS: Name: OSG1
; PARTS: Name: SV_Position
; PARTS: SystemValue: Position
; PARTS: Register: 0
; PARTS-NEXT: Mask: 15
; PARTS-NEXT: ExclusiveMask: 0
; PARTS: Name: A
; PARTS: Register: 1
; PARTS-NEXT: Mask: 1
; PARTS-NEXT: ExclusiveMask: 14
; PARTS: Name: B
; PARTS: Register: 1
; PARTS-NEXT: Mask: 6
; PARTS-NEXT: ExclusiveMask: 9
; PARTS: Name: PSV0
; PARTS: OutputPositionPresent: 1
; PARTS: SigInputVectors: 2
; PARTS: SigOutputVectors: [ 2, 0, 0, 0 ]
; PARTS: SigInputElements:
; PARTS: Name: POSITION
; PARTS: SigOutputElements:
; PARTS: Kind: Position
; PARTS: Name: B
; PARTS: StartRow: 1
; PARTS-NEXT: Cols: 2
; PARTS-NEXT: StartCol: 1
; PARTS: InputOutputMap:
