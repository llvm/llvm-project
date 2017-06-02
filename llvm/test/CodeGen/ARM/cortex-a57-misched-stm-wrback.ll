; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -misched-postra -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s
; N=3 STMIA_UPD should have latency 2cyc and writeback latency 1cyc

; CHECK:       ********** MI Scheduling **********
; We need second, post-ra scheduling to have STM instruction combined from single-stores
; CHECK:       ********** MI Scheduling **********
; CHECK:       schedule starting
; CHECK:       STMIA_UPD
; CHECK:       rdefs left
; CHECK-NEXT:  Latency            : 2
; CHECK:       Successors
; CHECK:       data
; CHECK-SAME:  Latency=1

define i32 @bar(i32 %v0, i32 %v1, i32 %v2, i32* %addr) {

  %addr.1 = getelementptr i32, i32* %addr, i32 0
  store i32 %v0, i32* %addr.1

  %addr.2 = getelementptr i32, i32* %addr, i32 1
  store i32 %v1, i32* %addr.2

  %addr.3 = getelementptr i32, i32* %addr, i32 2
  store i32 %v2, i32* %addr.3
  
  %ptr_after = getelementptr i32, i32* %addr, i32 3
  %val = ptrtoint i32* %ptr_after to i32
  
  %rv1 = mul i32 %val, %v0
  %rv2 = mul i32 %rv1, %v1
  %rv3 = mul i32 %rv2, %v2

  ret i32 %rv3
}

