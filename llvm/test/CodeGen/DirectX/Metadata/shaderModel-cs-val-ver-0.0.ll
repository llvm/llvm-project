; RUN: opt -S -dxil-prepare  %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

define void @entry() #0 {
entry:
  ret void
}

; Make sure experimental attribute is left when validation version is 0.0.
; CHECK:attributes #0 = { noinline nounwind "exp-shader"="cs" }
attributes #0 = { noinline nounwind "exp-shader"="cs" "hlsl.numthreads"="1,2,1" "hlsl.shader"="compute" }

!dx.valver = !{!0}

!0 = !{i32 0, i32 0}
