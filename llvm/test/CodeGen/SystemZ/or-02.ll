; Test 32-bit ORs in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the lowest useful OILL value.
define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: oill %r2, 1
; CHECK: br %r14
  %or = or i32 %a, 1
  ret i32 %or
}

; Check the high end of the OILL range.
define i32 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK: oill %r2, 65535
; CHECK: br %r14
  %or = or i32 %a, 65535
  ret i32 %or
}

; Check the lowest useful OILH range, which is the next value up.
define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: oilh %r2, 1
; CHECK: br %r14
  %or = or i32 %a, 65536
  ret i32 %or
}

; Check the lowest useful OILF value, which is the next value up again.
define i32 @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK: oilf %r2, 65537
; CHECK: br %r14
  %or = or i32 %a, 65537
  ret i32 %or
}

; Check the high end of the OILH range.
define i32 @f5(i32 %a) {
; CHECK-LABEL: f5:
; CHECK: oilh %r2, 65535
; CHECK: br %r14
  %or = or i32 %a, -65536
  ret i32 %or
}

; Check the next value up, which must use OILF instead.
define i32 @f6(i32 %a) {
; CHECK-LABEL: f6:
; CHECK: oilf %r2, 4294901761
; CHECK: br %r14
  %or = or i32 %a, -65535
  ret i32 %or
}

; Check the highest useful OILF value.
define i32 @f7(i32 %a) {
; CHECK-LABEL: f7:
; CHECK: oilf %r2, 4294967294
; CHECK: br %r14
  %or = or i32 %a, -2
  ret i32 %or
}
