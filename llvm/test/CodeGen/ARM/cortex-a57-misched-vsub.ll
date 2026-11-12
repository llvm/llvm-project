; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -misched-postra -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s

; CHECK-LABEL:  subv_i32:%bb.0
; CHECK:        SU(8): {{.*}} VSUBv4i32
; CHECK-NEXT:   # preds left
; CHECK-NEXT:   # succs left
; CHECK-NEXT:   # rdefs left
; CHECK-NEXT:   Latency : 3

define <4 x i32> @subv_i32(<4 x i32>, <4 x i32>) {
  %3 = sub <4 x i32> %1, %0
  ret <4 x i32> %3
}

; CHECK-LABEL:  subv_f32:%bb.0
; CHECK:        SU(8): {{.*}} VSUBfq
; CHECK-NEXT:   # preds left
; CHECK-NEXT:   # succs left
; CHECK-NEXT:   # rdefs left
; CHECK-NEXT:   Latency : 5

define <4 x float> @subv_f32(<4 x float>, <4 x float>) {
  %3 = fsub <4 x float> %0, %1
  ret <4 x float> %3
}
