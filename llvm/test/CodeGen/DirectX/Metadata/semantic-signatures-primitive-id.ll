; RUN: opt -disable-output -passes='print<dxil-signature>' %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
; RUN: opt -S -dxil-translate-metadata %s | FileCheck %s --check-prefix=MD

; SV_PrimitiveID is a system-generated pixel input. Unlike SV_SampleIndex, it is
; allocated like an arbitrary input, so it is packed at row 1 after A. The
; container parts are covered by ContainerData/SemanticSignatures-primitive-id.ll.
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

; ANALYSIS:      Inputs: 2 elements, 2 vectors
; ANALYSIS-NEXT: 0: A rows=1 cols=1 at 0:0 usage=1 dynamic=0
; ANALYSIS-NEXT: 1: SV_PrimitiveID rows=1 cols=1 at 1:0 usage=1 dynamic=0

; MD-DAG: !{i32 0, !"A", i8 9, i8 0, !{{[0-9]+}}, i8 2, i32 1, i8 1, i32 0, i8 0, !{{[0-9]+}}}
; MD-DAG: !{i32 1, !"SV_PrimitiveID", i8 5, i8 10, !{{[0-9]+}}, i8 1, i32 1, i8 1, i32 1, i8 0, !{{[0-9]+}}}
