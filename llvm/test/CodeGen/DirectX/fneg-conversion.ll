; RUN: llc %s --filetype=asm -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

define float @negateF(float %0) #0 {
; CHECK:  %2 = fsub float -0.000000e+00, %0
  %2 = fneg float %0
  ret float %2
}

define double @negateD(double %0) #0 {
; CHECK: %2 = fsub double -0.000000e+00, %0
  %2 = fneg double %0
  ret double %2
}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}