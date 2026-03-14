; Test 64-bit ORs in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the lowest useful OILL value.
define i64 @f1(i64 %a) {
; CHECK-LABEL: f1:
; CHECK: oill %r2, 1
; CHECK: br %r14
  %or = or i64 %a, 1
  ret i64 %or
}

; Check the high end of the OILL range.
define i64 @f2(i64 %a) {
; CHECK-LABEL: f2:
; CHECK: oill %r2, 65535
; CHECK: br %r14
  %or = or i64 %a, 65535
  ret i64 %or
}

; Check the lowest useful OILH value, which is the next value up.
define i64 @f3(i64 %a) {
; CHECK-LABEL: f3:
; CHECK: oilh %r2, 1
; CHECK: br %r14
  %or = or i64 %a, 65536
  ret i64 %or
}

; Check the lowest useful OILF value, which is the next value up again.
define i64 @f4(i64 %a) {
; CHECK-LABEL: f4:
; CHECK: oilf %r2, 4294901759
; CHECK: br %r14
  %or = or i64 %a, 4294901759
  ret i64 %or
}

; Check the high end of the OILH range.
define i64 @f5(i64 %a) {
; CHECK-LABEL: f5:
; CHECK: oilh %r2, 65535
; CHECK: br %r14
  %or = or i64 %a, 4294901760
  ret i64 %or
}

; Check the high end of the OILF range.
define i64 @f6(i64 %a) {
; CHECK-LABEL: f6:
; CHECK: oilf %r2, 4294967295
; CHECK: br %r14
  %or = or i64 %a, 4294967295
  ret i64 %or
}

; Check the lowest useful OIHL value, which is the next value up.
define i64 @f7(i64 %a) {
; CHECK-LABEL: f7:
; CHECK: oihl %r2, 1
; CHECK: br %r14
  %or = or i64 %a, 4294967296
  ret i64 %or
}

; Check the next value up again, which must use two ORs.
define i64 @f8(i64 %a) {
; CHECK-LABEL: f8:
; CHECK: oihl %r2, 1
; CHECK: oill %r2, 1
; CHECK: br %r14
  %or = or i64 %a, 4294967297
  ret i64 %or
}

; Check the high end of the OILL range.
define i64 @f9(i64 %a) {
; CHECK-LABEL: f9:
; CHECK: oihl %r2, 1
; CHECK: oill %r2, 65535
; CHECK: br %r14
  %or = or i64 %a, 4295032831
  ret i64 %or
}

; Check the next value up, which must use OILH
define i64 @f10(i64 %a) {
; CHECK-LABEL: f10:
; CHECK: oihl %r2, 1
; CHECK: oilh %r2, 1
; CHECK: br %r14
  %or = or i64 %a, 4295032832
  ret i64 %or
}

; Check the next value up again, which must use OILF
define i64 @f11(i64 %a) {
; CHECK-LABEL: f11:
; CHECK: oihl %r2, 1
; CHECK: oilf %r2, 65537
; CHECK: br %r14
  %or = or i64 %a, 4295032833
  ret i64 %or
}

; Check the high end of the OIHL range.
define i64 @f12(i64 %a) {
; CHECK-LABEL: f12:
; CHECK: oihl %r2, 65535
; CHECK: br %r14
  %or = or i64 %a, 281470681743360
  ret i64 %or
}

; Check a combination of the high end of the OIHL range and the high end
; of the OILF range.
define i64 @f13(i64 %a) {
; CHECK-LABEL: f13:
; CHECK: oihl %r2, 65535
; CHECK: oilf %r2, 4294967295
; CHECK: br %r14
  %or = or i64 %a, 281474976710655
  ret i64 %or
}

; Check the lowest useful OIHH value.
define i64 @f14(i64 %a) {
; CHECK-LABEL: f14:
; CHECK: oihh %r2, 1
; CHECK: br %r14
  %or = or i64 %a, 281474976710656
  ret i64 %or
}

; Check the next value up, which needs two ORs.
define i64 @f15(i64 %a) {
; CHECK-LABEL: f15:
; CHECK: oihh %r2, 1
; CHECK: oill %r2, 1
; CHECK: br %r14
  %or = or i64 %a, 281474976710657
  ret i64 %or
}

; Check the lowest useful OIHF value.
define i64 @f16(i64 %a) {
; CHECK-LABEL: f16:
; CHECK: oihf %r2, 65537
; CHECK: br %r14
  %or = or i64 %a, 281479271677952
  ret i64 %or
}

; Check the high end of the OIHH range.
define i64 @f17(i64 %a) {
; CHECK-LABEL: f17:
; CHECK: oihh %r2, 65535
; CHECK: br %r14
  %or = or i64 %a, 18446462598732840960
  ret i64 %or
}

; Check the high end of the OIHF range.
define i64 @f18(i64 %a) {
; CHECK-LABEL: f18:
; CHECK: oihf %r2, 4294967295
; CHECK: br %r14
  %or = or i64 %a, -4294967296
  ret i64 %or
}

; Check the highest useful OR value.
define i64 @f19(i64 %a) {
; CHECK-LABEL: f19:
; CHECK: oihf %r2, 4294967295
; CHECK: oilf %r2, 4294967294
; CHECK: br %r14
  %or = or i64 %a, -2
  ret i64 %or
}
