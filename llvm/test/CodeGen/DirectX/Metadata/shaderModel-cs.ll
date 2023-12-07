; RUN: opt -S -dxil-metadata-emit %s | FileCheck %s
target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"cs", i32 6, i32 6}

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.numthreads"="1,2,1" "hlsl.shader"="compute" }
