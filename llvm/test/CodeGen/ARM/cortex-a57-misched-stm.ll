; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -mattr=use-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s
; N=3 STMIB should have latency 2cyc

; CHECK:       ********** MI Scheduling **********
; We need second, post-ra scheduling to have STM instruction combined from single-stores
; CHECK:       ********** MI Scheduling **********
; CHECK:       schedule starting
; CHECK:       STMIB
; CHECK:       rdefs left
; CHECK-NEXT:  Latency            : 2

define i32 @test_stm(i32 %v0, i32 %v1, ptr %addr) {

  %addr.1 = getelementptr i32, ptr %addr, i32 1
  store i32 %v0, ptr %addr.1

  %addr.2 = getelementptr i32, ptr %addr, i32 2
  store i32 %v1, ptr %addr.2

  %addr.3 = getelementptr i32, ptr %addr, i32 3
  %val = ptrtoint ptr %addr to i32
  store i32 %val, ptr %addr.3

  %rv = add i32 %v0, %v1

  ret i32 %rv
}

