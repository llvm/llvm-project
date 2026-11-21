; RUN: llc -filetype=obj %s -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %s

; SV_PrimitiveID is written to the ISG1 and PSV0 parts as an allocated input
; element with the PrimitiveID system value. Packing is covered by
; Metadata/semantic-signatures-primitive-id.ll.
target triple = "dxil-pc-shadermodel6.0-pixel"

define void @main() #0 {
  %a = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  %pid = call i32 @llvm.dx.load.input.i32(i32 1, i32 0, i8 0, i32 poison)
  %f = uitofp i32 %pid to float
  %result = fadd float %a, %f
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
!4 = !{i32 1, !"SV_PrimitiveID", i32 5, i32 10, !7, i32 1, i32 1, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
!6 = !{i32 0, !"SV_Target", i32 9, i32 16, !7, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!7 = !{i32 0}

; CHECK: Name: ISG1
; CHECK:      Name: SV_PrimitiveID
; CHECK-NEXT: Index: 0
; CHECK-NEXT: SystemValue: PrimitiveID
; CHECK-NEXT: CompType: UInt32
; CHECK-NEXT: Register: 1
; CHECK-NEXT: Mask: 1
; CHECK-NEXT: ExclusiveMask: 1
; CHECK: Name: PSV0
; CHECK: SampleFrequency: 0
; CHECK: SigInputElements:
; CHECK:      Kind: Arbitrary
; CHECK:      Indices: [ 0 ]
; CHECK-NEXT: StartRow: 1
; CHECK-NEXT: Cols: 1
; CHECK-NEXT: StartCol: 0
; CHECK-NEXT: Allocated: true
; CHECK-NEXT: Kind: PrimitiveID
; CHECK-NEXT: ComponentType: UInt32
; CHECK-NEXT: Interpolation: Constant
