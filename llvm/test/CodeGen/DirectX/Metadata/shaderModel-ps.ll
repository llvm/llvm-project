
; RUN: opt -S -dxil-metadata-emit %s | FileCheck %s
; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
target triple = "dxil-pc-shadermodel5.0-pixel"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"ps", i32 5, i32 0}

; ANALYSIS: Shader Model Version : 5.0
; ANALYSIS: DXIL Version : 1.0
; ANALYSIS: Shader Stage : pixel

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="pixel" }
