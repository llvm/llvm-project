; RUN: opt -S -dxil-metadata-emit %s | FileCheck %s
target triple = "dxil-pc-shadermodel6.6-geometry"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"gs", i32 6, i32 6}

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="geometry" }
