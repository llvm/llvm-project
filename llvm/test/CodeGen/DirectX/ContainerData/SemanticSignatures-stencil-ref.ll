; RUN: llc -filetype=obj %s -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %s

; SV_StencilRef is written to the OSG1 and PSV0 parts as an unallocated output
; element with the StencilRef system value, and sets the StencilRef feature in
; SFI0. Analysis and metadata are covered by
; Metadata/semantic-signatures-stencil-ref.ll.
target triple = "dxil-pc-shadermodel6.0-pixel"

define void @main() #0 {
  %a = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  %s = fptoui float %a to i32
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %a)
  call void @llvm.dx.store.output.i32(i32 1, i32 0, i8 0, i32 %s)
  ret void
}
attributes #0 = { "hlsl.shader"="pixel" }

!dx.valver = !{!10}
!10 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3}
!2 = !{!5, !6}
!3 = !{i32 0, !"A", i32 9, i32 0, !7, i32 2, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!5 = !{i32 0, !"SV_Target", i32 9, i32 16, !7, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!6 = !{i32 1, !"SV_StencilRef", i32 5, i32 20, !7, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!7 = !{i32 0}

; CHECK: Name: SFI0
; CHECK: StencilRef: true
; CHECK: Name: OSG1
; CHECK:      Name: SV_StencilRef
; CHECK-NEXT: Index: 0
; CHECK-NEXT: SystemValue: StencilRef
; CHECK-NEXT: CompType: UInt32
; CHECK-NEXT: Register: 4294967295
; CHECK-NEXT: Mask: 1
; CHECK-NEXT: ExclusiveMask: 14
; CHECK: Name: PSV0
; CHECK: SigOutputElements:
; CHECK:      Kind: Target
; CHECK:      Indices: [ 0 ]
; CHECK-NEXT: StartRow: 0
; CHECK-NEXT: Cols: 1
; CHECK-NEXT: StartCol: 0
; CHECK-NEXT: Allocated: false
; CHECK-NEXT: Kind: StencilRef
; CHECK-NEXT: ComponentType: UInt32
