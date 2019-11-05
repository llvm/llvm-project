; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -mattr=use-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s
; 

@a = global i32 0, align 4
@b = global i32 0, align 4
@c = global i32 0, align 4

; CHECK:       ********** MI Scheduling **********
; We need second, post-ra scheduling to have LDM instruction combined from single-loads
; CHECK:       ********** MI Scheduling **********
; CHECK:       LDMIA_UPD
; CHECK:       rdefs left
; CHECK-NEXT:  Latency            : 4
; CHECK:       Successors:
; CHECK:       Data
; CHECK-SAME:  Latency=1
; CHECK-NEXT:  Data
; CHECK-SAME:  Latency=3
; CHECK-NEXT:  Data
; CHECK-SAME:  Latency=0
; CHECK-NEXT:  Data
; CHECK-SAME:  Latency=0
define i32 @bar(i32 %a1, i32 %b1, i32 %c1) minsize optsize {
  %1 = load i32, i32* @a, align 4
  %2 = load i32, i32* @b, align 4
  %3 = load i32, i32* @c, align 4

  %ptr_after = getelementptr i32, i32* @a, i32 3

  %ptr_val = ptrtoint i32* %ptr_after to i32
  %mul1 = mul i32 %ptr_val, %1
  %mul2 = mul i32 %mul1, %2
  %mul3 = mul i32 %mul2, %3
  ret i32 %mul3
}

