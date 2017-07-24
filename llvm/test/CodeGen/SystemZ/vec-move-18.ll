; Test insertions of memory values into 0 on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; Test VLLEZLF.
define <4 x i32> @f1(i32 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: vllezlf %v24, 0(%r2)
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %ret = insertelement <4 x i32> zeroinitializer, i32 %val, i32 0
  ret <4 x i32> %ret
}

; Test VLLEZLF with a float.
define <4 x float> @f2(float *%ptr) {
; CHECK-LABEL: f2:
; CHECK: vllezlf %v24, 0(%r2)
; CHECK: br %r14
  %val = load float, float *%ptr
  %ret = insertelement <4 x float> zeroinitializer, float %val, i32 0
  ret <4 x float> %ret
}

