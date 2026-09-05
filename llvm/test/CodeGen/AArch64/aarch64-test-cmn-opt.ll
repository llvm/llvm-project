; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs < %s | FileCheck %s

; This test file validates the CMN optimization for negative SUBS immediates. 
; Example: if (b > -34 && ...) should emit CMN w1, #34.

; Test: negative constant -34
define i32 @test_neg34(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test_neg34:
; CHECK:       cmn w1, #34
; CHECK:       ccmp w0, w2, #0, gt
; CHECK:       csel w0, w1, w0, lt
; CHECK:       ret
  %cmp1 = icmp sgt i32 %b, -34
  %cmp2 = icmp slt i32 %a, %c
  %and = and i1 %cmp1, %cmp2
  %res = select i1 %and, i32 %b, i32 %a
  ret i32 %res
}


; Test: boundary around CCMN immediate range
; -31 should not require the special transformation.
define i32 @test_neg31(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test_neg31:
; CHECK:       ccmp
; CHECK:       ret
  %cmp1 = icmp sgt i32 %b, -31
  %cmp2 = icmp slt i32 %a, %c
  %and = and i1 %cmp1, %cmp2
  %res = select i1 %and, i32 %b, i32 %a
  ret i32 %res
}


; Test: -32
define i32 @test_neg32(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test_neg32:
; CHECK:       ccmp
; CHECK:       ret
  %cmp1 = icmp sgt i32 %b, -32
  %cmp2 = icmp slt i32 %a, %c
  %and = and i1 %cmp1, %cmp2
  %res = select i1 %and, i32 %b, i32 %a
  ret i32 %res
}


; Test: -33
; This is the first value outside the CCMN immediate range.
define i32 @test_neg33(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test_neg33:
; CHECK:       cmn w1, #33
; CHECK:       ccmp
; CHECK:       ret
  %cmp1 = icmp sgt i32 %b, -33
  %cmp2 = icmp slt i32 %a, %c
  %and = and i1 %cmp1, %cmp2
  %res = select i1 %and, i32 %b, i32 %a
  ret i32 %res
}


; Test: -4095
; Maximum value representable by a 12-bit arithmetic immediate.
define i32 @test_neg4095(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test_neg4095:
; CHECK:       cmn w1, #4095
; CHECK:       ccmp
; CHECK:       ret
  %cmp1 = icmp sgt i32 %b, -4095
  %cmp2 = icmp slt i32 %a, %c
  %and = and i1 %cmp1, %cmp2
  %res = select i1 %and, i32 %b, i32 %a
  ret i32 %res
}


; Test: -4096
; Should not use plain CMN #4096 because it is outside
; the 12-bit immediate range.
define i32 @test_neg4096(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test_neg4096:
; CHECK-NOT:   cmn w1, #4096
; CHECK:       ret
  %cmp1 = icmp sgt i32 %b, -4096
  %cmp2 = icmp slt i32 %a, %c
  %and = and i1 %cmp1, %cmp2
  %res = select i1 %and, i32 %b, i32 %a
  ret i32 %res
}


; Test: positive constant.
; The CMN transformation should not apply.
define i32 @test_positive(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test_positive:
; CHECK-NOT:   cmn
; CHECK:       ret
  %cmp1 = icmp sgt i32 %b, 34
  %cmp2 = icmp slt i32 %a, %c
  %and = and i1 %cmp1, %cmp2
  %res = select i1 %and, i32 %b, i32 %a
  ret i32 %res
}
