; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7-arm-none-eabi"

; CHECK-LABEL: f:
; CHECK: vld1.64 {{.*}}, [r1:128]
; CHECK: .p2align 4
define void @f(<4 x i32>* %p) {
  store <4 x i32> <i32 -1, i32 0, i32 0, i32 -1>, <4 x i32>* %p, align 4
  ret void 
}

; CHECK-LABEL: f_optsize:
; CHECK: vld1.64 {{.*}}, [r1]
; CHECK: .p2align 3
define void @f_optsize(<4 x i32>* %p) optsize {
  store <4 x i32> <i32 -1, i32 0, i32 0, i32 -1>, <4 x i32>* %p, align 4
  ret void 
}
