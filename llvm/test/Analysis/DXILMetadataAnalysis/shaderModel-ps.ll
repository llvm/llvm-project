; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s
target triple = "dxil-pc-shadermodel5.0-pixel"

; CHECK: Shader Model Version : 5.0
; CHECK-NEXT: DXIL Version : 1.0
; CHECK-NEXT: Shader Stage : pixel
; CHECK-NEXT: Validator Version : 0
; CHECK-NEXT: entry
; CHECK-NEXT:  Function Shader Stage : pixel
; CHECK-NEXT:   NumThreads: 0,0,0
; CHECK-EMPTY:

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="pixel" }
