; RUN: opt -S -dxil-metadata-emit %s | FileCheck %s
target triple = "dxilv1.0-pc-shadermodel6.0-vertex"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"vs", i32 6, i32 0}

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="vertex" }
