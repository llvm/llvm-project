; RUN: not opt -S -dxil-translate-metadata %s 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-pixel"

; CHECK: Shader stage 'cs' for entry 'entry' different from specified target profile 'pixel'

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "exp-shader"="cs" "hlsl.numthreads"="1,2,1" "hlsl.shader"="compute" }
