; RUN: opt -S -dxil-metadata-emit %s | FileCheck %s
target triple = "dxil-pc-shadermodel6.8-compute"

; CHECK: !dx.version = !{![[DXVER:[0-9]+]]}
; CHECK: ![[DXVER]] = !{i32 1, i32 8}

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.numthreads"="1,2,1" "hlsl.shader"="compute" }
