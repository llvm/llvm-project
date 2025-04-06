; RUN: llc %s --filetype=asm -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

; CHECK: Function Attrs: nounwind memory(none)
; Function Attrs: norecurse nounwind readnone willreturn
define dso_local float @fma(float %0, float %1, float %2) local_unnamed_addr #0 {
  %4 = fmul float %0, %1
  %5 = fadd float %4, %2
  ret float %5
}

; CHECK: attributes #0 = { nounwind memory(none) }
; CHECK-NOT: attributes #

attributes #0 = { norecurse nounwind readnone willreturn "hlsl.export"}
