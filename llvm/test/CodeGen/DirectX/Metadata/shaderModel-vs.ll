; RUN: opt -S -dxil-translate-metadata %s | FileCheck %s
; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
target triple = "dxil-pc-shadermodel-vertex"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"vs", i32 0, i32 0}

; ANALYSIS: Shader Model Version : 0
; ANALYSIS-NEXT: DXIL Version : 1.0
; ANALYSIS-NEXT: Shader Stage : vertex
; ANALYSIS-NEXT: Validator Version : 0
; ANALYSIS-EMPTY:

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="vertex" }
