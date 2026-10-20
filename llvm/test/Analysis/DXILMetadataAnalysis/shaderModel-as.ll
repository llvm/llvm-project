; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s
target triple = "dxil-pc-shadermodel6-amplification"

; CHECK: Shader Model Version : 6
; CHECK-NEXT: DXIL Version : 1.0
; CHECK-NEXT: Target Shader Stage : amplification
; CHECK-NEXT: Validator Version : 0
; CHECK-NEXT: entry
; CHECK-NEXT:  Function Shader Stage : amplification
; CHECK-NEXT:   NumThreads: 0,0,0
; CHECK-EMPTY:

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="amplification" }
