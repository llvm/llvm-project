; Test 128-bit subtraction when the distinct-operands facility is available.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Test the case where both operands are in registers.
define i64 @f1(i64 %a, i64 %b, i64 %c, i64 %d, ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: slgrk %r2, %r4, %r5
; CHECK: slbgr
; CHECK: br %r14
  %x1 = insertelement <2 x i64> undef, i64 %b, i32 0
  %x2 = insertelement <2 x i64> %x1, i64 %c, i32 1
  %x = bitcast <2 x i64> %x2 to i128
  %y2 = insertelement <2 x i64> %x1, i64 %d, i32 1
  %y = bitcast <2 x i64> %y2 to i128
  %sub = sub i128 %x, %y
  %subv = bitcast i128 %sub to <2 x i64>
  %high = extractelement <2 x i64> %subv, i32 0
  store i64 %high, ptr %ptr
  %low = extractelement <2 x i64> %subv, i32 1
  ret i64 %low
}
