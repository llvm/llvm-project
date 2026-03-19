; RUN: opt -S -dxil-translate-metadata %s | FileCheck %s
target triple = "dxil-pc-shadermodel6-amplification"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"as", i32 6, i32 0}

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="amplification" }
