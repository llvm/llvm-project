; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s
target triple = "dxil-pc-shadermodel6.6-hull"

; CHECK: Shader Model Version : 6.6
; CHECK-NEXT: DXIL Version : 1.6
; CHECK-NEXT: Target Shader Stage : hull
; CHECK-NEXT: Validator Version : 0
; CHECK-NEXT: entry
; CHECK-NEXT:  Function Shader Stage : hull
; CHECK-NEXT:  NumThreads: 0,0,0
; CHECK-EMPTY:

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="hull" }
