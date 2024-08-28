; RUN: opt -S -dxil-translate-metadata %s | FileCheck %s
; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
target triple = "dxil-pc-shadermodel6.6-mesh"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"ms", i32 6, i32 6}

; ANALYSIS: Shader Model Version : 6.6
; ANALYSIS-NEXT: DXIL Version : 1.6
; ANALYSIS-NEXT: Shader Stage : mesh
; ANALYSIS-NEXT: Validator Version : 0
; ANALYSIS-EMPTY:

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="mesh" }
