; RUN: opt -S -dxil-intrinsic-expansion -dxil-translate-metadata %s | FileCheck %s --check-prefix=MD
; RUN: opt -S -passes='dxil-intrinsic-expansion,dxil-translate-metadata,dxil-op-lower,print<dxil-signature>' %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
; RUN: llc -O0 -filetype=obj %s -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %s --check-prefix=PARTS
; RUN: obj2yaml %t.dxbc | yaml2obj | obj2yaml | FileCheck %s --check-prefix=PARTS
; RUN: llvm-objcopy --dump-section=DXIL=%t.bc %t.dxbc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefixes=MD,OPS
; RUN: llc -O2 -filetype=obj %s -o - | obj2yaml | FileCheck %s --check-prefix=PARTS

target triple = "dxil-pc-shadermodel6.8-vertex"

define void @main() #0 {
  %p = call <4 x float> @llvm.dx.load.input.v4f32(i32 0, i32 0, i8 0, i32 poison)
  %uv = call <2 x float> @llvm.dx.load.input.v2f32(i32 1, i32 0, i8 0, i32 poison)
  %u = extractelement <2 x float> %uv, i32 0
  call void @llvm.dx.store.output.v4f32(i32 0, i32 0, i8 0, <4 x float> %p)
  call void @llvm.dx.store.output.f32(i32 1, i32 0, i8 0, float %u)
  call void @llvm.dx.store.output.v2f32(i32 2, i32 0, i8 0, <2 x float> %uv)
  ret void
}

attributes #0 = { "hlsl.shader"="vertex" }

!dx.valver = !{!20}
!20 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3, !4}
!2 = !{!5, !6, !7}
!3 = !{i32 0, !"POSITION", i32 9, i32 0, !10, i32 0, i32 1, i8 4, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!4 = !{i32 1, !"TEXCOORD", i32 9, i32 0, !10, i32 0, i32 1, i8 2, i32 -1, i8 -1, i8 0, i8 0, i32 0}
; The metadata keeps the source spelling, but OSG1 must use SV_Position.
!5 = !{i32 0, !"SV_POSITION", i32 9, i32 3, !10, i32 0, i32 1, i8 4, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!6 = !{i32 1, !"A", i32 9, i32 0, !10, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!7 = !{i32 2, !"B", i32 9, i32 0, !10, i32 0, i32 1, i8 2, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!10 = !{i32 0}

; ANALYSIS: Semantic signatures for 'main':
; ANALYSIS-NEXT: Inputs: 2 elements, 2 vectors
; ANALYSIS-NEXT: 0: POSITION rows=1 cols=4 at 0:0 usage=15 dynamic=0
; ANALYSIS-NEXT: 1: TEXCOORD rows=1 cols=2 at 1:0 usage=3 dynamic=0
; ANALYSIS-NEXT: Outputs: 3 elements, 2 vectors
; ANALYSIS-NEXT: 0: SV_POSITION rows=1 cols=4 at 0:0 usage=15 dynamic=0
; ANALYSIS-NEXT: 1: A rows=1 cols=1 at 1:0 usage=1 dynamic=0
; ANALYSIS-NEXT: 2: B rows=1 cols=2 at 1:1 usage=6 dynamic=0

; The packed location is row 1, col 1, but the access still uses ID 2, col 0.
; OPS: call void @dx.op.storeOutput.f32(i32 5, i32 2, i32 0, i8 0,

; MD-NOT: !dx.semantic.signatures
; MD: !dx.viewIdState = !{![[STATE:[0-9]+]]}
; MD: !dx.entryPoints = !{![[ENTRY:[0-9]+]]}
; MD-DAG: ![[STATE]] = !{[10 x i32] [i32 8, i32 8, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 0, i32 0]}
; MD-DAG: ![[ENTRY]] = !{ptr @main, !"main", ![[SIG:[0-9]+]], null, null}
; MD-DAG: ![[SIG]] = !{![[IN:[0-9]+]], ![[OUT:[0-9]+]], null}
; MD-DAG: ![[IN]] = !{![[POS:[0-9]+]], ![[UV:[0-9]+]]}
; MD-DAG: ![[OUT]] = !{![[SVPOS:[0-9]+]], ![[A:[0-9]+]], ![[B:[0-9]+]]}
; MD-DAG: ![[POS]] = !{i32 0, !"POSITION", i8 9, i8 0, ![[INDEX:[0-9]+]], i8 0, i32 1, i8 4, i32 0, i8 0, ![[USE4:[0-9]+]]}
; MD-DAG: ![[UV]] = !{i32 1, !"TEXCOORD", i8 9, i8 0, ![[INDEX]], i8 0, i32 1, i8 2, i32 1, i8 0, ![[USE2:[0-9]+]]}
; MD-DAG: ![[SVPOS]] = !{i32 0, !"SV_POSITION", i8 9, i8 3, ![[INDEX]], i8 0, i32 1, i8 4, i32 0, i8 0, ![[USE4]]}
; MD-DAG: ![[A]] = !{i32 1, !"A", i8 9, i8 0, ![[INDEX]], i8 0, i32 1, i8 1, i32 1, i8 0, ![[USE1:[0-9]+]]}
; MD-DAG: ![[B]] = !{i32 2, !"B", i8 9, i8 0, ![[INDEX]], i8 0, i32 1, i8 2, i32 1, i8 1, ![[USE2]]}
; MD-DAG: ![[USE4]] = !{i32 3, i32 15}
; MD-DAG: ![[USE2]] = !{i32 3, i32 3}
; MD-DAG: ![[USE1]] = !{i32 3, i32 1}
; MD-NOT: !dx.semantic.signatures

; PARTS: Name: ISG1
; PARTS: Name: POSITION
; PARTS: SystemValue: Undefined
; PARTS-NEXT: CompType: Float32
; PARTS-NEXT: Register: 0
; PARTS-NEXT: Mask: 15
; PARTS-NEXT: ExclusiveMask: 15
; PARTS: Name: TEXCOORD
; PARTS: Register: 1
; PARTS-NEXT: Mask: 3
; PARTS-NEXT: ExclusiveMask: 3
; PARTS: Name: OSG1
; PARTS: Name: SV_Position
; PARTS: SystemValue: Position
; PARTS: Register: 0
; PARTS-NEXT: Mask: 15
; PARTS-NEXT: ExclusiveMask: 0
; PARTS: Name: A
; PARTS: Register: 1
; PARTS-NEXT: Mask: 1
; PARTS-NEXT: ExclusiveMask: 14
; PARTS: Name: B
; PARTS: Register: 1
; PARTS-NEXT: Mask: 6
; PARTS-NEXT: ExclusiveMask: 9
; PARTS: Name: PSV0
; PARTS: OutputPositionPresent: 1
; PARTS: SigInputVectors: 2
; PARTS: SigOutputVectors: [ 2, 0, 0, 0 ]
; PARTS: SigInputElements:
; PARTS: Name: POSITION
; PARTS: SigOutputElements:
; PARTS: Kind: Position
; PARTS: Name: B
; PARTS: StartRow: 1
; PARTS-NEXT: Cols: 2
; PARTS-NEXT: StartCol: 1
; PARTS: InputOutputMap:
