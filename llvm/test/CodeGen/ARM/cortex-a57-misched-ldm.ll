; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -misched-postra -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s

; CHECK:       ********** MI Scheduling **********
; We need second, post-ra scheduling to have LDM instruction combined from single-loads
; CHECK:       ********** MI Scheduling **********
; CHECK:       LDMIA
; CHECK:       rdefs left
; CHECK-NEXT:  Latency            : 3
; CHECK:       Successors:
; CHECK:       data
; CHECK-SAME:  Latency=3
; CHECK-NEXT:  data 
; CHECK-SAME:  Latency=3

define i32 @foo(i32* %a) nounwind optsize {
entry:
  %b = getelementptr i32, i32* %a, i32 1
  %c = getelementptr i32, i32* %a, i32 2 
  %0 = load i32, i32* %a, align 4
  %1 = load i32, i32* %b, align 4
  %2 = load i32, i32* %c, align 4

  %mul1 = mul i32 %0, %1
  %mul2 = mul i32 %mul1, %2
  ret i32 %mul2
}

