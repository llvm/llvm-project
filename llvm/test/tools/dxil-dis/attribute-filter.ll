; RUN: llc %s --filetype=obj -o - | dxil-dis -o - | FileCheck %s

; CHECK: target triple = "dxil-ms-dx"
target triple = "dxil-unknown-shadermodel6.7-library"

; CHECK: Function Attrs: nounwind readnone
; Function Attrs: norecurse nounwind readnone willreturn
define float @fma(float %0, float %1, float %2) #0 {
  %4 = fmul float %0, %1
  %5 = fadd float %4, %2
  ret float %5
}

; CHECK: Function Attrs: readnone
; Function Attrs: norecurse readnone willreturn
define float @fma2(float %0, float %1, float %2) #1 {
  %4 = fmul float %0, %1
  %5 = fadd float %4, %2
  ret float %5
}

; CHECK: attributes #0 = { nounwind readnone "disable-tail-calls"="false" }
attributes #0 = { norecurse nounwind readnone willreturn "disable-tail-calls"="false" }

; CHECK: attributes #1 = { readnone "disable-tail-calls"="false" }
attributes #1 = { norecurse memory(none) willreturn "disable-tail-calls"="false" }
