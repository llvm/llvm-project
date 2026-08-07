; RUN: split-file %s %t
; RUN: not opt -S --dxil-translate-metadata %t/low-sm.ll 2>&1 | FileCheck %t/low-sm.ll
; RUN: not opt -S --dxil-translate-metadata %t/low-sm-for-range.ll 2>&1 | FileCheck %t/low-sm-for-range.ll

; Test that wavesize metadata is only allowed on applicable shader model versions

;--- low-sm.ll

; CHECK: Shader model 6.6 or greater is required to specify the "hlsl.wavesize" function attribute

target triple = "dxil-unknown-shadermodel6.5-compute"

define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.wavesize"="16,0,0" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

;--- low-sm-for-range.ll

; CHECK: Shader model 6.8 or greater is required to specify wave size range values of the "hlsl.wavesize" function attribute

target triple = "dxil-unknown-shadermodel6.7-compute"

define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.wavesize"="16,32,0" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
