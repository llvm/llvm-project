; RUN: not opt -S  -S -dxil-translate-metadata %s 2>&1 | FileCheck %s
target triple = "dxil-pc-shadermodel6.8-compute"

; CHECK: Non-library shader: One and only one entry expected

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
