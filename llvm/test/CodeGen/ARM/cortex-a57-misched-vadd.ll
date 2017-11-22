; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -misched-postra -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s

; CHECK-LABEL:  addv_i32:BB#0
; CHECK:        SU(8): {{.*}} VADDv4i32
; CHECK-NEXT:   # preds left
; CHECK-NEXT:   # succs left
; CHECK-NEXT:   # rdefs left
; CHECK-NEXT:   Latency : 3

define <4 x i32> @addv_i32(<4 x i32>, <4 x i32>) {
  %3 = add <4 x i32> %1, %0
  ret <4 x i32> %3
}

; CHECK-LABEL:  addv_f32:BB#0
; CHECK:        SU(8): {{.*}} VADDfq
; CHECK-NEXT:   # preds left
; CHECK-NEXT:   # succs left
; CHECK-NEXT:   # rdefs left
; CHECK-NEXT:   Latency : 5

define <4 x float> @addv_f32(<4 x float>, <4 x float>) {
  %3 = fadd <4 x float> %0, %1
  ret <4 x float> %3
}
