; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s
target triple = "dxil-pc-shadermodel6.8-library"

; CHECK: Shader Model Version : 6.8
; CHECK-NEXT: DXIL Version : 1.8
; CHECK-NEXT: Target Shader Stage : library
; CHECK-NEXT: Validator Version : 0
; CHECK-NEXT: entry_as
; CHECK-NEXT:   Function Shader Stage : amplification
; CHECK-NEXT:   NumThreads: 0,0,0
; CHECK-NEXT: entry_ms
; CHECK-NEXT:   Function Shader Stage : mesh
; CHECK-NEXT:   NumThreads: 0,0,0
; CHECK-NEXT: entry_cs
; CHECK-NEXT:   Function Shader Stage : compute
; CHECK-NEXT:   NumThreads: 1,2,1
; CHECK-EMPTY:

define void @entry_as() #0 {
entry:
  ret void
}

define i32 @entry_ms(i32 %a) #1 {
entry:
  ret i32 %a
}

define float @entry_cs(float %f) #3 {
entry:
  ret float %f
}

attributes #0 = { noinline nounwind "hlsl.shader"="amplification" }
attributes #1 = { noinline nounwind "hlsl.shader"="mesh" }
attributes #3 = { noinline nounwind "hlsl.numthreads"="1,2,1" "hlsl.shader"="compute" }
