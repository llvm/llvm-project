; RUN: llc -filetype=obj %s -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %s --check-prefix=PARTS
; RUN: llvm-objcopy --dump-section=DXIL=%t.bc %t.dxbc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=MD

; A vertex shader writing SV_RenderTargetArrayIndex feeds the rasterizer, which
; requires the ViewportAndRTArrayIndexFromAnyShaderFeedingRasterizer feature.
target triple = "dxil-pc-shadermodel6.0-vertex"

define void @main() #0 {
  %p = call <4 x float> @llvm.dx.load.input.v4f32(i32 0, i32 0, i8 0, i32 poison)
  call void @llvm.dx.store.output.v4f32(i32 0, i32 0, i8 0, <4 x float> %p)
  call void @llvm.dx.store.output.i32(i32 1, i32 0, i8 0, i32 1)
  ret void
}
attributes #0 = { "hlsl.shader"="vertex" }

!dx.valver = !{!10}
!10 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3}
!2 = !{!4, !5}
!3 = !{i32 0, !"POSITION", i32 9, i32 0, !7, i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
!4 = !{i32 0, !"SV_Position", i32 9, i32 3, !7, i32 4, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
!5 = !{i32 1, !"SV_RenderTargetArrayIndex", i32 5, i32 4, !7, i32 1, i32 1, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
!7 = !{i32 0}

; MD-DAG: !{i32 1, !"SV_RenderTargetArrayIndex", i8 5, i8 4, !{{[0-9]+}}, i8 1, i32 1, i8 1, i32 1, i8 0, !{{[0-9]+}}}
; MD-DAG: !{i32 0, i64 512}

; PARTS: Name: SFI0
; PARTS: ViewportAndRTArrayIndexFromAnyShaderFeedingRasterizer: true
; PARTS: Name: OSG1
; PARTS:      Name: SV_RenderTargetArrayIndex
; PARTS-NEXT: Index: 0
; PARTS-NEXT: SystemValue: RenderTargetArrayIndex
; PARTS-NEXT: CompType: UInt32
; PARTS-NEXT: Register: 1
; PARTS-NEXT: Mask: 1
