; RUN: split-file %s %t
; RUN: llc -filetype=obj %t/uint.ll -o %t/uint.dxbc
; RUN: obj2yaml %t/uint.dxbc | FileCheck %s --check-prefixes=CHECK,U32
; RUN: llc -filetype=obj %t/int.ll -o %t/int.dxbc
; RUN: obj2yaml %t/int.dxbc | FileCheck %s --check-prefixes=CHECK,I32

; SV_IsFrontFace is written to the ISG1 and PSV0 parts as an allocated input
; element with the IsFrontFace system value. A bool or uint is carried as
; UInt32 and an int as SInt32. Packing is covered by
; Metadata/semantic-signatures-is-front-face.ll.

;--- uint.ll
target triple = "dxil-pc-shadermodel6.0-pixel"

define void @main() #0 {
  %a = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  %ff = call i32 @llvm.dx.load.input.i32(i32 1, i32 0, i8 0, i32 poison)
  %f = uitofp i32 %ff to float
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
!4 = !{i32 1, !"SV_IsFrontFace", i32 5, i32 13, !7, i32 1, i32 1, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
!6 = !{i32 0, !"SV_Target", i32 9, i32 16, !7, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!7 = !{i32 0}

;--- int.ll
target triple = "dxil-pc-shadermodel6.0-pixel"

define void @main() #0 {
  %a = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  %ff = call i32 @llvm.dx.load.input.i32(i32 1, i32 0, i8 0, i32 poison)
  %f = sitofp i32 %ff to float
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
!4 = !{i32 1, !"SV_IsFrontFace", i32 4, i32 13, !7, i32 1, i32 1, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
!6 = !{i32 0, !"SV_Target", i32 9, i32 16, !7, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!7 = !{i32 0}

; The test runs once per file. U32 and I32 differ only in the component type.
; CHECK: Name: ISG1
; CHECK:      Name: SV_IsFrontFace
; CHECK-NEXT: Index: 0
; CHECK-NEXT: SystemValue: IsFrontFace
; U32-NEXT:   CompType: UInt32
; I32-NEXT:   CompType: SInt32
; CHECK-NEXT: Register: 1
; CHECK-NEXT: Mask: 1
; CHECK-NEXT: ExclusiveMask: 1
; CHECK: Name: PSV0
; CHECK: SampleFrequency: 0
; CHECK: SigInputElements:
; The first Kind is element A, which comes before SV_IsFrontFace.
; CHECK:      Kind: Arbitrary
; CHECK:      Indices: [ 0 ]
; CHECK-NEXT: StartRow: 1
; CHECK-NEXT: Cols: 1
; CHECK-NEXT: StartCol: 0
; CHECK-NEXT: Allocated: true
; CHECK-NEXT: Kind: IsFrontFace
; U32-NEXT:   ComponentType: UInt32
; I32-NEXT:   ComponentType: SInt32
; CHECK-NEXT: Interpolation: Constant
