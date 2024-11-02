; RUN: opt -S -passes=dxil-translate-metadata %s | FileCheck %s
; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
target triple = "dxil-pc-shadermodel6.8-compute"

; CHECK: !dx.version = !{![[DXVER:[0-9]+]]}
; CHECK: ![[DXVER]] = !{i32 1, i32 8}

; ANALYSIS: Shader Model Version : 6.8
; ANALYSIS-NEXT: DXIL Version : 1.8
; ANALYSIS-NEXT: Shader Stage : compute
; ANALYSIS-NEXT: Validator Version : 0
; ANALYSIS-EMPTY:

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.numthreads"="1,2,1" "hlsl.shader"="compute" }
