; Test vector multiply-and-add on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)

; Test a v4f32 multiply-and-add.
define <4 x float> @f1(<4 x float> %dummy, <4 x float> %val1,
                       <4 x float> %val2, <4 x float> %val3) {
; CHECK-LABEL: f1:
; CHECK: vfmasb %v24, %v26, %v28, %v30
; CHECK: br %r14
  %ret = call <4 x float> @llvm.fma.v4f32 (<4 x float> %val1,
                                           <4 x float> %val2,
                                           <4 x float> %val3)
  ret <4 x float> %ret
}

; Test a v4f32 multiply-and-subtract.
define <4 x float> @f2(<4 x float> %dummy, <4 x float> %val1,
                       <4 x float> %val2, <4 x float> %val3) {
; CHECK-LABEL: f2:
; CHECK: vfmssb %v24, %v26, %v28, %v30
; CHECK: br %r14
  %negval3 = fneg <4 x float> %val3
  %ret = call <4 x float> @llvm.fma.v4f32 (<4 x float> %val1,
                                           <4 x float> %val2,
                                           <4 x float> %negval3)
  ret <4 x float> %ret
}
