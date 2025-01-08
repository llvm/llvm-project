; RUN: llc %s --filetype=obj -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

define float @fma(float %0, float %1, float %2) #0 {
  ; verify reassoc and contract are converted to fast
  ; CHECK: %4 = fmul fast float %0, %1
  %4 = fmul reassoc float %0, %1
  ; CHECK-NEXT: %5 = fadd fast float %4, %2
  %5 = fadd contract float %4, %2
  ; verify these are converted to a single fast flag
  ; CHECK-NEXT: %6 = fmul fast float %0, %1
  %6 = fmul reassoc contract float %0, %1
  ; verify these flags are maintained
  ; CHECK-NEXT: %7 = fadd nnan ninf nsz arcp float %0, %1
  %7 = fadd nnan ninf nsz arcp float %0, %1
  ; verify that afn is removed
  ; CHECK-NEXT: %8 = fmul float %0, %1
  %8 = fmul afn float %0, %1
  ret float %5
}

attributes #0 = { norecurse nounwind readnone willreturn "disable-tail-calls"="false" "waveops-include-helper-lanes" "fp32-denorm-mode"="any" "hlsl.export" }

