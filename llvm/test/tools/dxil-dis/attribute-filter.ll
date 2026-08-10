; RUN: llc %s --filetype=obj -o - | dxil-dis -o - | FileCheck %s

; CHECK: target triple = "dxil-ms-dx"
target triple = "dxil-unknown-shadermodel6.7-library"

; CHECK: Function Attrs: nounwind readnone
define float @fma(float %0, float %1, float %2) #0 {
  %4 = fmul float %0, %1
  %5 = fadd float %4, %2
  ret float %5
}

; CHECK: Function Attrs: readnone
define float @fma2(float %0, float %1, float %2) #1 {
  %4 = fmul float %0, %1
  %5 = fadd float %4, %2
  ret float %5
}

; CHECK: attributes #0 = { nounwind readnone "fp32-denorm-mode"="any" "waveops-include-helper-lanes" }
attributes #0 = { norecurse nounwind readnone willreturn "disable-tail-calls"="false" "waveops-include-helper-lanes" "fp32-denorm-mode"="any" "hlsl.export" }

; CHECK: attributes #1 = { readnone "fp32-denorm-mode"="ftz" "waveops-include-helper-lanes" }
attributes #1 = { norecurse memory(none) willreturn "disable-tail-calls"="false" "waveops-include-helper-lanes" "fp32-denorm-mode"="ftz" "hlsl.export" }
