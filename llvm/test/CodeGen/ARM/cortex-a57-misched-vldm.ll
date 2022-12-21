; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -mattr=use-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s

; CHECK:       ********** MI Scheduling **********
; We need second, post-ra scheduling to have VLDM instruction combined from single-loads
; CHECK:       ********** MI Scheduling **********
; CHECK:       VLDMDIA
; CHECK:       rdefs left
; CHECK-NEXT:  Latency            : 6
; CHECK:       Successors:
; CHECK:       Data
; CHECK-SAME:  Latency=5
; CHECK-NEXT:  Data
; CHECK-SAME:  Latency=0
; CHECK-NEXT:  Data
; CHECK-SAME:  Latency=0

define double @foo(ptr %a) nounwind optsize {
entry:
  %b = getelementptr double, ptr %a, i32 1
  %c = getelementptr double, ptr %a, i32 2 
  %0 = load double, ptr %a, align 4
  %1 = load double, ptr %b, align 4
  %2 = load double, ptr %c, align 4

  %mul1 = fmul double %0, %1
  %mul2 = fmul double %mul1, %2
  ret double %mul2
}

