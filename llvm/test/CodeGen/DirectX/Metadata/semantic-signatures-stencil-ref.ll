; RUN: opt -disable-output -passes='print<dxil-signature>' %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
; RUN: opt -S -dxil-op-lower %s | FileCheck %s --check-prefix=DXIL
; RUN: opt -S -dxil-translate-metadata %s | FileCheck %s --check-prefix=MD

; SV_StencilRef is an unpacked pixel shader output: it is written with
; storeOutput, but has no output row and no entry in the dependency table. It
; requires the StencilRef shader flag. The container parts are covered by
; ContainerData/SemanticSignatures-stencil-ref.ll.
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

; ANALYSIS:      Outputs: 2 elements, 1 vectors
; ANALYSIS-NEXT: 0: SV_Target rows=1 cols=1 at 0:0 usage=1 dynamic=0
; ANALYSIS-NEXT: 1: SV_StencilRef rows=1 cols=1 at 4294967295:255 usage=1 dynamic=0

; DXIL: call void @dx.op.storeOutput.i32(i32 5, i32 1, i32 0, i8 0, i32 %{{.*}})

; MD-DAG: !dx.viewIdState = !{![[STATE:[0-9]+]]}
; MD-DAG: ![[STATE]] = !{[6 x i32] [i32 4, i32 4, i32 1, i32 0, i32 0, i32 0]}
; MD-DAG: !{i32 1, !"SV_StencilRef", i8 5, i8 20, !{{[0-9]+}}, i8 0, i32 1, i8 1, i32 -1, i8 -1, !{{[0-9]+}}}
; MD-DAG: !{i32 0, i64 2048}
