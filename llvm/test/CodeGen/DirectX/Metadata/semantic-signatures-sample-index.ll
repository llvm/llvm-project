; RUN: opt -disable-output -passes='print<dxil-signature>' %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
; RUN: opt -S -dxil-translate-metadata %s | FileCheck %s --check-prefix=MD

; SV_SampleIndex is an unpacked input element: it keeps signature ID 1, but is
; not allocated a row or column, so the arbitrary input A is still packed at
; row 0.
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

; ANALYSIS:      Inputs: 2 elements, 1 vectors
; ANALYSIS-NEXT: 0: A rows=1 cols=1 at 0:0 usage=1 dynamic=0
; ANALYSIS-NEXT: 1: SV_SampleIndex rows=1 cols=1 at 4294967295:255 usage=0 dynamic=0
; ANALYSIS-NEXT: Outputs: 1 elements, 1 vectors

; The DXIL signature element has a start row of -1 and start column of -1.
; MD-DAG: !{i32 0, !"A", i8 9, i8 0, !{{[0-9]+}}, i8 2, i32 1, i8 1, i32 0, i8 0, !{{[0-9]+}}}
; MD-DAG: !{i32 1, !"SV_SampleIndex", i8 5, i8 12, !{{[0-9]+}}, i8 1, i32 1, i8 1, i32 -1, i8 -1, null}
