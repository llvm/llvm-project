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

; Test VLLEZLF with a float when the result is stored to memory.
define void @f3(float *%ptr, <4 x float> *%res) {
; CHECK-LABEL: f3:
; CHECK: vllezlf [[REG:%v[0-9]+]], 0(%r2)
; CHECK: vst [[REG]], 0(%r3)
; CHECK: br %r14
  %val = load float, float *%ptr
  %ret = insertelement <4 x float> zeroinitializer, float %val, i32 0
  store <4 x float> %ret, <4 x float> *%res
  ret void
}

