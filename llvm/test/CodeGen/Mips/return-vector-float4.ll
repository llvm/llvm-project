; RUN: llc -march=mipsel -mattr=+android < %s | FileCheck %s

define <4 x float> @retvec4() nounwind readnone {
entry:
; CHECK: lwc1 $f0
; CHECK: lwc1 $f2
; CHECK: lwc1 $f1
; CHECK: lwc1 $f3

  ret <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>
}

