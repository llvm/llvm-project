; Test insertions of i32s into the low half of an i64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Insertion of an i32 can be done using LR.
define i64 @f1(i64 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK-NOT: {{%r[23]}}
; CHECK: lr %r2, %r3
; CHECK: br %r14
  %low = zext i32 %b to i64
  %high = and i64 %a, -4294967296
  %res = or i64 %high, %low
  ret i64 %res
}

; ... and again with the operands reversed.
define i64 @f2(i64 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK-NOT: {{%r[23]}}
; CHECK: lr %r2, %r3
; CHECK: br %r14
  %low = zext i32 %b to i64
  %high = and i64 %a, -4294967296
  %res = or i64 %low, %high
  ret i64 %res
}

; Like f1, but with "in register" zero extension.
define i64 @f3(i64 %a, i64 %b) {
; CHECK-LABEL: f3:
; CHECK-NOT: {{%r[23]}}
; CHECK: lr %r2, %r3
; CHECK: br %r14
  %low = and i64 %b, 4294967295
  %high = and i64 %a, -4294967296
  %res = or i64 %high, %low
  ret i64 %res
}

; ... and again with the operands reversed.
define i64 @f4(i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK-NOT: {{%r[23]}}
; CHECK: lr %r2, %r3
; CHECK: br %r14
  %low = and i64 %b, 4294967295
  %high = and i64 %a, -4294967296
  %res = or i64 %low, %high
  ret i64 %res
}

; Unary operations can be done directly into the low half.
define i64 @f5(i64 %a, i32 %b) {
; CHECK-LABEL: f5:
; CHECK-NOT: {{%r[23]}}
; CHECK: lcr %r2, %r3
; CHECK: br %r14
  %neg = sub i32 0, %b
  %low = zext i32 %neg to i64
  %high = and i64 %a, -4294967296
  %res = or i64 %high, %low
  ret i64 %res
}

; ...likewise three-operand binary operations like RLL.
define i64 @f6(i64 %a, i32 %b) {
; CHECK-LABEL: f6:
; CHECK-NOT: {{%r[23]}}
; CHECK: rll %r2, %r3, 1
; CHECK: br %r14
  %parta = shl i32 %b, 1
  %partb = lshr i32 %b, 31
  %rot = or i32 %parta, %partb
  %low = zext i32 %rot to i64
  %high = and i64 %a, -4294967296
  %res = or i64 %low, %high
  ret i64 %res
}

; Loads can be done directly into the low half.  The range of L is checked
; in the move tests.
define i64 @f7(i64 %a, i32 *%src) {
; CHECK-LABEL: f7:
; CHECK-NOT: {{%r[23]}}
; CHECK: l %r2, 0(%r3)
; CHECK: br %r14
  %b = load i32 *%src
  %low = zext i32 %b to i64
  %high = and i64 %a, -4294967296
  %res = or i64 %high, %low
  ret i64 %res
}

; ...likewise extending loads.
define i64 @f8(i64 %a, i8 *%src) {
; CHECK-LABEL: f8:
; CHECK-NOT: {{%r[23]}}
; CHECK: lb %r2, 0(%r3)
; CHECK: br %r14
  %byte = load i8 *%src
  %b = sext i8 %byte to i32
  %low = zext i32 %b to i64
  %high = and i64 %a, -4294967296
  %res = or i64 %high, %low
  ret i64 %res
}

; Check a case like f1 in which there is no AND.  We simply know from context
; that the upper half of one OR operand and the lower half of the other are
; both clear.
define i64 @f9(i64 %a, i32 %b) {
; CHECK-LABEL: f9:
; CHECK: sllg %r2, %r2, 32
; CHECK: lr %r2, %r3
; CHECK: br %r14
  %shift = shl i64 %a, 32
  %low = zext i32 %b to i64
  %or = or i64 %shift, %low
  ret i64 %or
}

; ...and again with the operands reversed.
define i64 @f10(i64 %a, i32 %b) {
; CHECK-LABEL: f10:
; CHECK: sllg %r2, %r2, 32
; CHECK: lr %r2, %r3
; CHECK: br %r14
  %shift = shl i64 %a, 32
  %low = zext i32 %b to i64
  %or = or i64 %low, %shift
  ret i64 %or
}

; Like f9, but with "in register" zero extension.
define i64 @f11(i64 %a, i64 %b) {
; CHECK-LABEL: f11:
; CHECK: lr %r2, %r3
; CHECK: br %r14
  %shift = shl i64 %a, 32
  %low = and i64 %b, 4294967295
  %or = or i64 %shift, %low
  ret i64 %or
}

; ...and again with the operands reversed.
define i64 @f12(i64 %a, i64 %b) {
; CHECK-LABEL: f12:
; CHECK: lr %r2, %r3
; CHECK: br %r14
  %shift = shl i64 %a, 32
  %low = and i64 %b, 4294967295
  %or = or i64 %low, %shift
  ret i64 %or
}

; Like f9, but for larger shifts than 32.
define i64 @f13(i64 %a, i32 %b) {
; CHECK-LABEL: f13:
; CHECK: sllg %r2, %r2, 60
; CHECK: lr %r2, %r3
; CHECK: br %r14
  %shift = shl i64 %a, 60
  %low = zext i32 %b to i64
  %or = or i64 %shift, %low
  ret i64 %or
}

; We previously wrongly removed the upper AND as dead.
define i64 @f14(i64 %a, i64 %b) {
; CHECK-LABEL: f14:
; CHECK: risbg {{%r[0-5]}}, %r2, 6, 134, 0
; CHECK: br %r14
  %and1 = and i64 %a, 144115188075855872
  %and2 = and i64 %b, 15
  %or = or i64 %and1, %and2
  %res = icmp eq i64 %or, 0
  %ext = sext i1 %res to i64
  ret i64 %ext
}
