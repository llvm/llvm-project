; RUN: split-file %s %t
; RUN: opt -disable-output -passes='print<dxil-signature>' %t/uint.ll 2>&1 | FileCheck %s --check-prefix=ANALYSIS
; RUN: opt -disable-output -passes='print<dxil-signature>' %t/int.ll 2>&1 | FileCheck %s --check-prefix=ANALYSIS
; RUN: opt -S -dxil-translate-metadata %t/uint.ll | FileCheck %s --check-prefix=MD-UINT
; RUN: opt -S -dxil-translate-metadata %t/int.ll | FileCheck %s --check-prefix=MD-INT

; SV_IsFrontFace is a system-generated pixel input. Like SV_PrimitiveID, it is
; allocated like an arbitrary input, so it is packed at row 1 after A. A bool or
; uint is carried as U32 (5) and an int as I32 (4). All of them
; are read with the i32 load overload.

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

; MD-UINT-DAG: !{i32 0, !"A", i8 9, i8 0, !{{[0-9]+}}, i8 2, i32 1, i8 1, i32 0, i8 0, !{{[0-9]+}}}
; MD-UINT-DAG: !{i32 1, !"SV_IsFrontFace", i8 5, i8 13, !{{[0-9]+}}, i8 1, i32 1, i8 1, i32 1, i8 0, !{{[0-9]+}}}

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

; MD-INT-DAG: !{i32 0, !"A", i8 9, i8 0, !{{[0-9]+}}, i8 2, i32 1, i8 1, i32 0, i8 0, !{{[0-9]+}}}
; MD-INT-DAG: !{i32 1, !"SV_IsFrontFace", i8 4, i8 13, !{{[0-9]+}}, i8 1, i32 1, i8 1, i32 1, i8 0, !{{[0-9]+}}}

; ANALYSIS:      Inputs: 2 elements, 2 vectors
; ANALYSIS-NEXT: 0: A rows=1 cols=1 at 0:0 usage=1 dynamic=0
; ANALYSIS-NEXT: 1: SV_IsFrontFace rows=1 cols=1 at 1:0 usage=1 dynamic=0
