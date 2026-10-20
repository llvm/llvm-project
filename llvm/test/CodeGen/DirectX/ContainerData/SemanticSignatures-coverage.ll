; RUN: opt -S -dxil-translate-metadata %s | FileCheck %s --check-prefix=MD
; RUN: opt -disable-output -passes='print<dxil-signature>' %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
; RUN: llc -filetype=obj %s -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %s --check-prefix=PARTS

; SV_Coverage is read through dx.op.coverage, so it is not in the input
; signature. As an output it is part of the signature, but it is never
; allocated a register location.

target triple = "dxil-pc-shadermodel6.8-pixel"

define void @main() #0 {
  %c = call i32 @llvm.dx.coverage()
  %color = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %color)
  call void @llvm.dx.store.output.i32(i32 1, i32 0, i8 0, i32 %c)
  ret void
}
attributes #0 = { "hlsl.shader"="pixel" }

!dx.valver = !{!20}
!20 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3}
!2 = !{!4, !5}
!3 = !{i32 0, !"A", i32 9, i32 0, !10, i32 2, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!4 = !{i32 0, !"SV_Target", i32 9, i32 16, !10, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!5 = !{i32 1, !"SV_Coverage", i32 5, i32 14, !10, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!10 = !{i32 0}

; ANALYSIS: Inputs: 1 elements, 1 vectors
; ANALYSIS-NEXT: 0: A rows=1 cols=1 at 0:0 usage=1 dynamic=0
; ANALYSIS-NEXT: Outputs: 2 elements, 1 vectors
; ANALYSIS-NEXT: 0: SV_Target rows=1 cols=1 at 0:0 usage=1 dynamic=0
; ANALYSIS-NEXT: 1: SV_Coverage rows=1 cols=1 at 4294967295:255 usage=1 dynamic=0

; The not-packed element keeps the unallocated start row and column, and its
; usage mask is relative to the first component.
; MD: !dx.viewIdState = !{![[STATE:[0-9]+]]}
; MD-DAG: ![[STATE]] = !{[6 x i32] [i32 4, i32 4, i32 1, i32 0, i32 0, i32 0]}
; MD-DAG: !{i32 1, !"SV_Coverage", i8 5, i8 14, !{{[0-9]+}}, i8 0, i32 1, i8 1, i32 -1, i8 -1, ![[WRITE:[0-9]+]]}
; MD-DAG: ![[WRITE]] = !{i32 3, i32 1}

; PARTS: Name: ISG1
; PARTS: Name: A
; PARTS-NOT: Name: SV_Coverage
; PARTS: Name: OSG1
; PARTS: Name: SV_Target
; PARTS: Register: 0
; PARTS-NEXT: Mask: 1
; PARTS-NEXT: ExclusiveMask: 14
; PARTS: Name: SV_Coverage
; PARTS-NEXT: Index: 0
; PARTS-NEXT: SystemValue: Coverage
; PARTS-NEXT: CompType: UInt32
; PARTS-NEXT: Register: 4294967295
; PARTS-NEXT: Mask: 1
; PARTS-NEXT: ExclusiveMask: 14
; PARTS: Name: PSV0
; PARTS: SigInputVectors: 1
; PARTS-NEXT: SigOutputVectors: [ 1, 0, 0, 0 ]
; PARTS: SigOutputElements:
; PARTS: Kind: Target
; PARTS: Indices: [ 0 ]
; PARTS-NEXT: StartRow: 0
; PARTS-NEXT: Cols: 1
; PARTS-NEXT: StartCol: 0
; PARTS-NEXT: Allocated: false
; PARTS-NEXT: Kind: Coverage
; PARTS-NEXT: ComponentType: UInt32
; PARTS-NEXT: Interpolation: Undefined
; PARTS: InputOutputMap:
; PARTS-NEXT: - [ 0x1, 0x0, 0x0, 0x0 ]
