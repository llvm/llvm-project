; RUN: opt -S -dxil-metadata-emit %s | FileCheck %s
; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
target triple = "dxil-pc-shadermodel6.0-vertex"

; CHECK: !dx.version = !{![[DXVER:[0-9]+]]}
; CHECK: ![[DXVER]] = !{i32 1, i32 0}

; ANALYSIS: Shader Model Version : 6.0
; ANALYSIS: DXIL Version : 1.0
; ANALYSIS: Shader Stage : vertex

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="vertex" }
