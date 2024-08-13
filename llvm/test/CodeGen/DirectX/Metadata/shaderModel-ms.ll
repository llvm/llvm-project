; RUN: opt -S -dxil-metadata-emit %s | FileCheck %s
; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
target triple = "dxil-pc-shadermodel6.6-mesh"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"ms", i32 6, i32 6}

; ANALYSIS: Shader Model Version : 6.6
; ANALYSIS: DXIL Version : 1.6
; ANALYSIS: Shader Stage : mesh

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="mesh" }
