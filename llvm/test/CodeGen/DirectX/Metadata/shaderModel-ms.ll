; RUN: opt -S -dxil-translate-metadata %s | FileCheck %s
target triple = "dxil-pc-shadermodel6.6-mesh"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"ms", i32 6, i32 6}

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="mesh" }
