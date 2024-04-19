; RUN: opt -S -dxil-metadata-emit %s | FileCheck %s
target triple = "dxilv1.7-pc-shadermodel6.7-amplification"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"as", i32 6, i32 7}

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="amplification" }
