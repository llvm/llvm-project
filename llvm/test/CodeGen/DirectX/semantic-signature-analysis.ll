; RUN: opt -disable-output -passes='print<dxil-signature>' %s 2>&1 | FileCheck %s

; Exercise the analysis independently of metadata translation and container
; emission. A library's entries must use their own stages and signature IDs.
target triple = "dxil-pc-shadermodel6.8-library"

define void @vs() #0 {
  %a = call <3 x float> @llvm.dx.load.input.v3f32(i32 0, i32 0, i8 0, i32 poison)
  %b = call float @llvm.dx.load.input.f32(i32 1, i32 0, i8 0, i32 poison)
  call void @llvm.dx.store.output.v3f32(i32 0, i32 0, i8 0, <3 x float> %a)
  call void @llvm.dx.store.output.f32(i32 1, i32 0, i8 0, float %b)
  ret void
}

define void @ps() #1 {
  %a = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 1, i32 poison)
  %i = fptoui float %a to i32
  %row = and i32 %i, 1
  %b = call float @llvm.dx.load.input.f32(i32 1, i32 %row, i8 0, i32 poison)
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 2, float %b)
  ret void
}

attributes #0 = { "hlsl.shader"="vertex" }
attributes #1 = { "hlsl.shader"="pixel" }

!dx.semantic.signatures = !{!0, !1}
!0 = !{ptr @vs, !2, !2}
!1 = !{ptr @ps, !3, !4}
!2 = !{!5, !6}
!3 = !{!5, !7}
!4 = !{!8}
!5 = !{i32 0, !"A", i32 9, i32 0, !9, i32 0, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!6 = !{i32 1, !"B", i32 9, i32 0, !9, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!7 = !{i32 1, !"B", i32 9, i32 0, !10, i32 0, i32 2, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!8 = !{i32 0, !"SV_Target", i32 9, i32 16, !11, i32 0, i32 1, i8 4, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!9 = !{i32 0}
!10 = !{i32 0, i32 1}
!11 = !{i32 4}

; CHECK: Semantic signatures for 'vs':
; CHECK-NEXT: Inputs: 2 elements, 2 vectors
; CHECK-NEXT: 0: A rows=1 cols=3 at 0:0 usage=7 dynamic=0
; CHECK-NEXT: 1: B rows=1 cols=1 at 1:0 usage=1 dynamic=0
; CHECK-NEXT: Outputs: 2 elements, 1 vectors
; CHECK-NEXT: 0: A rows=1 cols=3 at 0:0 usage=7 dynamic=0
; CHECK-NEXT: 1: B rows=1 cols=1 at 0:3 usage=8 dynamic=0
; CHECK-NEXT: Semantic signatures for 'ps':
; CHECK-NEXT: Inputs: 2 elements, 2 vectors
; CHECK-NEXT: 0: A rows=1 cols=3 at 0:0 usage=2 dynamic=0
; CHECK-NEXT: 1: B rows=2 cols=1 at 0:3 usage=8 dynamic=1
; CHECK-NEXT: Outputs: 1 elements, 5 vectors
; CHECK-NEXT: 0: SV_Target rows=1 cols=4 at 4:0 usage=4 dynamic=0
