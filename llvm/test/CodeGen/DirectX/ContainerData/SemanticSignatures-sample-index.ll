; RUN: llc -filetype=obj %s -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %s

; SV_SampleIndex is written to the ISG1 and PSV0 parts as an unallocated input
; element, and marks the pixel shader as sample-frequency. Packing is covered by
; Metadata/semantic-signatures-sample-index.ll and the dx.op.sampleIndex
; lowering by sample_index.ll.
target triple = "dxil-pc-shadermodel6.0-pixel"

define void @main() #0 {
  %p = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  %si = call i32 @llvm.dx.sample.index()
  %f = uitofp i32 %si to float
  %result = fadd float %p, %f
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %result)
  ret void
}
attributes #0 = { "hlsl.shader"="pixel" }

!dx.valver = !{!10}
!10 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3, !4}
!2 = !{!6}
!3 = !{i32 0, !"A", i32 9, i32 0, !7, i32 2, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!4 = !{i32 1, !"SV_SampleIndex", i32 5, i32 12, !7, i32 1, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!6 = !{i32 0, !"SV_Target", i32 9, i32 16, !7, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!7 = !{i32 0}

; CHECK: Name: ISG1
; CHECK:      Name: SV_SampleIndex
; CHECK-NEXT: Index: 0
; CHECK-NEXT: SystemValue: SampleIndex
; CHECK-NEXT: CompType: UInt32
; CHECK-NEXT: Register: 4294967295
; CHECK-NEXT: Mask: 1
; CHECK-NEXT: ExclusiveMask: 0
; CHECK: Name: PSV0
; CHECK: SampleFrequency: 1
; CHECK:      SigInputVectors: 1
; CHECK: SigInputElements:
; CHECK:      Kind: Arbitrary
; CHECK:      Indices: [ 0 ]
; CHECK-NEXT: StartRow: 0
; CHECK-NEXT: Cols: 1
; CHECK-NEXT: StartCol: 0
; CHECK-NEXT: Allocated: false
; CHECK-NEXT: Kind: SampleIndex
; CHECK-NEXT: ComponentType: UInt32
; CHECK-NEXT: Interpolation: Constant
