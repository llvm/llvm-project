; Test load and zero rightmost byte.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Check LZRF with no displacement.
define i32 @f1(i32 *%src) {
; CHECK-LABEL: f1:
; CHECK: lzrf %r2, 0(%r2)
; CHECK: br %r14
  %val = load i32, i32 *%src
  %and = and i32 %val, 4294967040
  ret i32 %and
}

; Check the high end of the LZRF range.
define i32 @f2(i32 *%src) {
; CHECK-LABEL: f2:
; CHECK: lzrf %r2, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131071
  %val = load i32, i32 *%ptr
  %and = and i32 %val, 4294967040
  ret i32 %and
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f3(i32 *%src) {
; CHECK-LABEL: f3:
; CHECK: agfi %r2, 524288
; CHECK: lzrf %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131072
  %val = load i32, i32 *%ptr
  %and = and i32 %val, 4294967040
  ret i32 %and
}

; Check the high end of the negative LZRF range.
define i32 @f4(i32 *%src) {
; CHECK-LABEL: f4:
; CHECK: lzrf %r2, -4(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -1
  %val = load i32, i32 *%ptr
  %and = and i32 %val, 4294967040
  ret i32 %and
}

; Check the low end of the LZRF range.
define i32 @f5(i32 *%src) {
; CHECK-LABEL: f5:
; CHECK: lzrf %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131072
  %val = load i32, i32 *%ptr
  %and = and i32 %val, 4294967040
  ret i32 %and
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f6(i32 *%src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, -524292
; CHECK: lzrf %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131073
  %val = load i32, i32 *%ptr
  %and = and i32 %val, 4294967040
  ret i32 %and
}

; Check that LZRF allows an index.
define i32 @f7(i64 %src, i64 %index) {
; CHECK-LABEL: f7:
; CHECK: lzrf %r2, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i32 *
  %val = load i32 , i32 *%ptr
  %and = and i32 %val, 4294967040
  ret i32 %and
}

; Check LZRG with no displacement.
define i64 @f8(i64 *%src) {
; CHECK-LABEL: f8:
; CHECK: lzrg %r2, 0(%r2)
; CHECK: br %r14
  %val = load i64, i64 *%src
  %and = and i64 %val, 18446744073709551360
  ret i64 %and
}

; Check the high end of the LZRG range.
define i64 @f9(i64 *%src) {
; CHECK-LABEL: f9:
; CHECK: lzrg %r2, 524280(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65535
  %val = load i64, i64 *%ptr
  %and = and i64 %val, 18446744073709551360
  ret i64 %and
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f10(i64 *%src) {
; CHECK-LABEL: f10:
; CHECK: agfi %r2, 524288
; CHECK: lzrg %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65536
  %val = load i64, i64 *%ptr
  %and = and i64 %val, 18446744073709551360
  ret i64 %and
}

; Check the high end of the negative LZRG range.
define i64 @f11(i64 *%src) {
; CHECK-LABEL: f11:
; CHECK: lzrg %r2, -8(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -1
  %val = load i64, i64 *%ptr
  %and = and i64 %val, 18446744073709551360
  ret i64 %and
}

; Check the low end of the LZRG range.
define i64 @f12(i64 *%src) {
; CHECK-LABEL: f12:
; CHECK: lzrg %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65536
  %val = load i64, i64 *%ptr
  %and = and i64 %val, 18446744073709551360
  ret i64 %and
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f13(i64 *%src) {
; CHECK-LABEL: f13:
; CHECK: agfi %r2, -524296
; CHECK: lzrg %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65537
  %val = load i64, i64 *%ptr
  %and = and i64 %val, 18446744073709551360
  ret i64 %and
}

; Check that LZRG allows an index.
define i64 @f14(i64 %src, i64 %index) {
; CHECK-LABEL: f14:
; CHECK: lzrg %r2, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i64 *
  %val = load i64 , i64 *%ptr
  %and = and i64 %val, 18446744073709551360
  ret i64 %and
}

; Check LLZRGF with no displacement.
define i64 @f15(i32 *%src) {
; CHECK-LABEL: f15:
; CHECK: llzrgf %r2, 0(%r2)
; CHECK: br %r14
  %val = load i32, i32 *%src
  %ext = zext i32 %val to i64
  %and = and i64 %ext, 18446744073709551360
  ret i64 %and
}

; ... and the other way around.
define i64 @f16(i32 *%src) {
; CHECK-LABEL: f16:
; CHECK: llzrgf %r2, 0(%r2)
; CHECK: br %r14
  %val = load i32, i32 *%src
  %and = and i32 %val, 4294967040
  %ext = zext i32 %and to i64
  ret i64 %ext
}

; Check the high end of the LLZRGF range.
define i64 @f17(i32 *%src) {
; CHECK-LABEL: f17:
; CHECK: llzrgf %r2, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131071
  %val = load i32, i32 *%ptr
  %and = and i32 %val, 4294967040
  %ext = zext i32 %and to i64
  ret i64 %ext
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f18(i32 *%src) {
; CHECK-LABEL: f18:
; CHECK: agfi %r2, 524288
; CHECK: llzrgf %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131072
  %val = load i32, i32 *%ptr
  %and = and i32 %val, 4294967040
  %ext = zext i32 %and to i64
  ret i64 %ext
}

; Check the high end of the negative LLZRGF range.
define i64 @f19(i32 *%src) {
; CHECK-LABEL: f19:
; CHECK: llzrgf %r2, -4(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -1
  %val = load i32, i32 *%ptr
  %and = and i32 %val, 4294967040
  %ext = zext i32 %and to i64
  ret i64 %ext
}

; Check the low end of the LLZRGF range.
define i64 @f20(i32 *%src) {
; CHECK-LABEL: f20:
; CHECK: llzrgf %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131072
  %val = load i32, i32 *%ptr
  %and = and i32 %val, 4294967040
  %ext = zext i32 %and to i64
  ret i64 %ext
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f21(i32 *%src) {
; CHECK-LABEL: f21:
; CHECK: agfi %r2, -524292
; CHECK: llzrgf %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131073
  %val = load i32, i32 *%ptr
  %and = and i32 %val, 4294967040
  %ext = zext i32 %and to i64
  ret i64 %ext
}

; Check that LLZRGF allows an index.
define i64 @f22(i64 %src, i64 %index) {
; CHECK-LABEL: f22:
; CHECK: llzrgf %r2, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i32 *
  %val = load i32 , i32 *%ptr
  %and = and i32 %val, 4294967040
  %ext = zext i32 %and to i64
  ret i64 %ext
}

; Check that we still get a RISBGN if the source is in a register.
define i64 @f23(i32 %src) {
; CHECK-LABEL: f23:
; CHECK: risbgn %r2, %r2, 32, 183, 0
; CHECK: br %r14
  %and = and i32 %src, 4294967040
  %ext = zext i32 %and to i64
  ret i64 %ext
}

