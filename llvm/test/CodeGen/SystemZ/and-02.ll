; Test 32-bit ANDs in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; ANDs with 1 can use NILF.
define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: nilf %r2, 1
; CHECK: br %r14
  %and = and i32 %a, 1
  ret i32 %and
}

; ...but RISBLG is available as a three-address form.
define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: risblg %r2, %r3, 31, 159, 0
; CHECK: br %r14
  %and = and i32 %b, 1
  ret i32 %and
}

; ...same for 4.
define i32 @f3(i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: risblg %r2, %r3, 29, 157, 0
; CHECK: br %r14
  %and = and i32 %b, 4
  ret i32 %and
}

; ANDs with 5 must use NILF.
define i32 @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK: nilf %r2, 5
; CHECK: br %r14
  %and = and i32 %a, 5
  ret i32 %and
}

; ...a single RISBLG isn't enough.
define i32 @f5(i32 %a, i32 %b) {
; CHECK-LABEL: f5:
; CHECK-NOT: risb
; CHECK: br %r14
  %and = and i32 %b, 5
  ret i32 %and
}

; Check the highest 16-bit constant that must be handled by NILF.
define i32 @f6(i32 %a) {
; CHECK-LABEL: f6:
; CHECK: nilf %r2, 65533
; CHECK: br %r14
  %and = and i32 %a, 65533
  ret i32 %and
}

; ...a single RISBLG isn't enough.
define i32 @f7(i32 %a, i32 %b) {
; CHECK-LABEL: f7:
; CHECK-NOT: risb
; CHECK: br %r14
  %and = and i32 %b, 65533
  ret i32 %and
}

; Check the next highest value, which can use NILF.
define i32 @f8(i32 %a) {
; CHECK-LABEL: f8:
; CHECK: nilf %r2, 65534
; CHECK: br %r14
  %and = and i32 %a, 65534
  ret i32 %and
}

; ...although the three-address case should use RISBLG.
define i32 @f9(i32 %a, i32 %b) {
; CHECK-LABEL: f9:
; CHECK: risblg %r2, %r3, 16, 158, 0
; CHECK: br %r14
  %and = and i32 %b, 65534
  ret i32 %and
}

; ANDs of 0xffff are zero extensions from i16.
define i32 @f10(i32 %a, i32 %b) {
; CHECK-LABEL: f10:
; CHECK: llhr %r2, %r3
; CHECK: br %r14
  %and = and i32 %b, 65535
  ret i32 %and
}

; Check the next value up, which must again use NILF.
define i32 @f11(i32 %a) {
; CHECK-LABEL: f11:
; CHECK: nilf %r2, 65536
; CHECK: br %r14
  %and = and i32 %a, 65536
  ret i32 %and
}

; ...but the three-address case can use RISBLG.
define i32 @f12(i32 %a, i32 %b) {
; CHECK-LABEL: f12:
; CHECK: risblg %r2, %r3, 15, 143, 0
; CHECK: br %r14
  %and = and i32 %b, 65536
  ret i32 %and
}

; Check the lowest useful NILH value.
define i32 @f13(i32 %a) {
; CHECK-LABEL: f13:
; CHECK: nilh %r2, 1
; CHECK: br %r14
  %and = and i32 %a, 131071
  ret i32 %and
}

; ...but RISBLG is OK in the three-address case.
define i32 @f14(i32 %a, i32 %b) {
; CHECK-LABEL: f14:
; CHECK: risblg %r2, %r3, 15, 159, 0
; CHECK: br %r14
  %and = and i32 %b, 131071
  ret i32 %and
}

; Check the highest useful NILF value.
define i32 @f15(i32 %a) {
; CHECK-LABEL: f15:
; CHECK: nilf %r2, 4294901758
; CHECK: br %r14
  %and = and i32 %a, -65538
  ret i32 %and
}

; Check the next value up, which is the highest useful NILH value.
define i32 @f16(i32 %a) {
; CHECK-LABEL: f16:
; CHECK: nilh %r2, 65534
; CHECK: br %r14
  %and = and i32 %a, -65537
  ret i32 %and
}

; Check the next value up, which is the first useful NILL value.
define i32 @f17(i32 %a) {
; CHECK-LABEL: f17:
; CHECK: nill %r2, 0
; CHECK: br %r14
  %and = and i32 %a, -65536
  ret i32 %and
}

; ...although the three-address case should use RISBLG.
define i32 @f18(i32 %a, i32 %b) {
; CHECK-LABEL: f18:
; CHECK: risblg %r2, %r3, 0, 143, 0
; CHECK: br %r14
  %and = and i32 %b, -65536
  ret i32 %and
}

; Check the next value up again, which can still use NILL.
define i32 @f19(i32 %a) {
; CHECK-LABEL: f19:
; CHECK: nill %r2, 1
; CHECK: br %r14
  %and = and i32 %a, -65535
  ret i32 %and
}

; Check the next value up again, which cannot use RISBLG.
define i32 @f20(i32 %a, i32 %b) {
; CHECK-LABEL: f20:
; CHECK-NOT: risb
; CHECK: br %r14
  %and = and i32 %b, -65534
  ret i32 %and
}

; Check the last useful mask, which can use NILL.
define i32 @f21(i32 %a) {
; CHECK-LABEL: f21:
; CHECK: nill %r2, 65534
; CHECK: br %r14
  %and = and i32 %a, -2
  ret i32 %and
}

; ...or RISBLG for the three-address case.
define i32 @f22(i32 %a, i32 %b) {
; CHECK-LABEL: f22:
; CHECK: risblg %r2, %r3, 0, 158, 0
; CHECK: br %r14
  %and = and i32 %b, -2
  ret i32 %and
}

; Test that RISBLG can be used when inserting a non-wraparound mask
; into another register.
define i64 @f23(i64 %a, i32 %b) {
; CHECK-LABEL: f23:
; CHECK: risblg %r2, %r3, 30, 158, 0
; CHECK: br %r14
  %and1 = and i64 %a, -4294967296
  %and2 = and i32 %b, 2
  %ext = zext i32 %and2 to i64
  %or = or i64 %and1, %ext
  ret i64 %or
}

; ...and when inserting a wrap-around mask.
define i64 @f24(i64 %a, i32 %b) {
; CHECK-LABEL: f24:
; CHECK: risblg %r2, %r3, 30, 156
; CHECK: br %r14
  %and1 = and i64 %a, -4294967296
  %and2 = and i32 %b, -5
  %ext = zext i32 %and2 to i64
  %or = or i64 %and1, %ext
  ret i64 %or
}
