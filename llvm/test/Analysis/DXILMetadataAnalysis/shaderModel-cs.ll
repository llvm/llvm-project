; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: Shader Model Version : 6.6
; CHECK-NEXT: DXIL Version : 1.6
; CHECK-NEXT: Target Shader Stage : compute
; CHECK-NEXT: Validator Version : 0
; CHECK-NEXT: entry
; CHECK-NEXT:  Function Shader Stage : compute
; CHECK-NEXT:   NumThreads: 1,2,1
; CHECK-EMPTY:

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "exp-shader"="cs" "hlsl.numthreads"="1,2,1" "hlsl.shader"="compute" }
