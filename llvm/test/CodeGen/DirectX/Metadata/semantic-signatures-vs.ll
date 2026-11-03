; RUN: split-file %s %t
; RUN: opt -S -dxil-intrinsic-expansion -dxil-translate-metadata %t/shader.ll | FileCheck %s
; RUN: opt -S -passes='dxil-intrinsic-expansion,dxil-translate-metadata,dxil-op-lower,print<dxil-signature>' %t/shader.ll 2>&1 | FileCheck %s --check-prefix=ANALYSIS
; RUN: opt -S -passes=dxil-translate-metadata %t/unused-masks.ll | FileCheck %s --check-prefix=UNUSED
; RUN: llc -O0 -filetype=obj %t/shader.ll -o %t.dxbc
; RUN: llvm-objcopy --dump-section=DXIL=%t.bc %t.dxbc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefixes=CHECK,OPS

;--- shader.ll
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
!3 = !{i32 0, !"POSITION", i32 9, i32 0, !10, i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
!4 = !{i32 1, !"TEXCOORD", i32 9, i32 0, !10, i32 0, i32 1, i8 2, i32 1, i8 0, i8 0, i8 0, i32 0}
!5 = !{i32 0, !"SV_POSITION", i32 9, i32 3, !10, i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
!6 = !{i32 1, !"A", i32 9, i32 0, !10, i32 0, i32 1, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
!7 = !{i32 2, !"B", i32 9, i32 0, !10, i32 0, i32 1, i8 2, i32 1, i8 1, i8 0, i8 0, i32 0}
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

; CHECK-NOT: !dx.semantic.signatures
; CHECK: !dx.viewIdState = !{![[STATE:[0-9]+]]}
; CHECK: !dx.entryPoints = !{![[ENTRY:[0-9]+]]}
; CHECK-DAG: ![[STATE]] = !{[10 x i32] [i32 8, i32 8, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 0, i32 0]}
; CHECK-DAG: ![[ENTRY]] = !{ptr @main, !"main", ![[SIG:[0-9]+]], null, null}
; CHECK-DAG: ![[SIG]] = !{![[IN:[0-9]+]], ![[OUT:[0-9]+]], null}
; CHECK-DAG: ![[IN]] = !{![[POS:[0-9]+]], ![[UV:[0-9]+]]}
; CHECK-DAG: ![[OUT]] = !{![[SVPOS:[0-9]+]], ![[A:[0-9]+]], ![[B:[0-9]+]]}
; CHECK-DAG: ![[POS]] = !{i32 0, !"POSITION", i8 9, i8 0, ![[INDEX:[0-9]+]], i8 0, i32 1, i8 4, i32 0, i8 0, ![[USE4:[0-9]+]]}
; CHECK-DAG: ![[UV]] = !{i32 1, !"TEXCOORD", i8 9, i8 0, ![[INDEX]], i8 0, i32 1, i8 2, i32 1, i8 0, ![[USE2:[0-9]+]]}
; CHECK-DAG: ![[SVPOS]] = !{i32 0, !"SV_POSITION", i8 9, i8 3, ![[INDEX]], i8 0, i32 1, i8 4, i32 0, i8 0, ![[USE4]]}
; CHECK-DAG: ![[A]] = !{i32 1, !"A", i8 9, i8 0, ![[INDEX]], i8 0, i32 1, i8 1, i32 1, i8 0, ![[USE1:[0-9]+]]}
; CHECK-DAG: ![[B]] = !{i32 2, !"B", i8 9, i8 0, ![[INDEX]], i8 0, i32 1, i8 2, i32 1, i8 1, ![[USE2]]}
; CHECK-DAG: ![[USE4]] = !{i32 3, i32 15}
; CHECK-DAG: ![[USE2]] = !{i32 3, i32 3}
; CHECK-DAG: ![[USE1]] = !{i32 3, i32 1}
; CHECK-NOT: !dx.semantic.signatures

;--- unused-masks.ll
; Stale masks must be cleared on unused elements, not copied into DXIL metadata.
; UNUSED: !{i32 0, !"A", i8 9, i8 0, !{{[0-9]+}}, i8 0, i32 1, i8 1, i32 0, i8 0, null}
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !1}
!1 = !{!2}
!2 = !{i32 0, !"A", i32 9, i32 0, !3, i32 0, i32 1, i8 1, i32 0, i8 0, i8 15, i8 15, i32 0}
!3 = !{i32 0}
