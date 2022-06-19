; Test zero extensions from a halfword to an i32.  The tests here
; assume z10 register pressure, without the high words being available.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Test register extension, starting with an i32.
define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: llhr %r2, %r2
; CHECK: br %r14
  %half = trunc i32 %a to i16
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; ...and again with an i64.
define i32 @f2(i64 %a) {
; CHECK-LABEL: f2:
; CHECK: llhr %r2, %r2
; CHECK: br %r14
  %half = trunc i64 %a to i16
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Check ANDs that are equivalent to zero extension.
define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: llhr %r2, %r2
; CHECK: br %r14
  %ext = and i32 %a, 65535
  ret i32 %ext
}

; Check LLH with no displacement.
define i32 @f4(i16 *%src) {
; CHECK-LABEL: f4:
; CHECK: llh %r2, 0(%r2)
; CHECK: br %r14
  %half = load i16, i16 *%src
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Check the high end of the LLH range.
define i32 @f5(i16 *%src) {
; CHECK-LABEL: f5:
; CHECK: llh %r2, 524286(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 262143
  %half = load i16, i16 *%ptr
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Check the next halfword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f6(i16 *%src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, 524288
; CHECK: llh %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 262144
  %half = load i16, i16 *%ptr
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Check the high end of the negative LLH range.
define i32 @f7(i16 *%src) {
; CHECK-LABEL: f7:
; CHECK: llh %r2, -2(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -1
  %half = load i16, i16 *%ptr
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Check the low end of the LLH range.
define i32 @f8(i16 *%src) {
; CHECK-LABEL: f8:
; CHECK: llh %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -262144
  %half = load i16, i16 *%ptr
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Check the next halfword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f9(i16 *%src) {
; CHECK-LABEL: f9:
; CHECK: agfi %r2, -524290
; CHECK: llh %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -262145
  %half = load i16, i16 *%ptr
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Check that LLH allows an index
define i32 @f10(i64 %src, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: llh %r2, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i16 *
  %half = load i16, i16 *%ptr
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Test a case where we spill the source of at least one LLHR.  We want
; to use LLH if possible.
define void @f11(i32 *%ptr) {
; CHECK-LABEL: f11:
; CHECK: llh {{%r[0-9]+}}, 186(%r15)
; CHECK: br %r14
  %val0 = load volatile i32, i32 *%ptr
  %val1 = load volatile i32, i32 *%ptr
  %val2 = load volatile i32, i32 *%ptr
  %val3 = load volatile i32, i32 *%ptr
  %val4 = load volatile i32, i32 *%ptr
  %val5 = load volatile i32, i32 *%ptr
  %val6 = load volatile i32, i32 *%ptr
  %val7 = load volatile i32, i32 *%ptr
  %val8 = load volatile i32, i32 *%ptr
  %val9 = load volatile i32, i32 *%ptr
  %val10 = load volatile i32, i32 *%ptr
  %val11 = load volatile i32, i32 *%ptr
  %val12 = load volatile i32, i32 *%ptr
  %val13 = load volatile i32, i32 *%ptr
  %val14 = load volatile i32, i32 *%ptr
  %val15 = load volatile i32, i32 *%ptr

  %trunc0 = trunc i32 %val0 to i16
  %trunc1 = trunc i32 %val1 to i16
  %trunc2 = trunc i32 %val2 to i16
  %trunc3 = trunc i32 %val3 to i16
  %trunc4 = trunc i32 %val4 to i16
  %trunc5 = trunc i32 %val5 to i16
  %trunc6 = trunc i32 %val6 to i16
  %trunc7 = trunc i32 %val7 to i16
  %trunc8 = trunc i32 %val8 to i16
  %trunc9 = trunc i32 %val9 to i16
  %trunc10 = trunc i32 %val10 to i16
  %trunc11 = trunc i32 %val11 to i16
  %trunc12 = trunc i32 %val12 to i16
  %trunc13 = trunc i32 %val13 to i16
  %trunc14 = trunc i32 %val14 to i16
  %trunc15 = trunc i32 %val15 to i16

  %ext0 = zext i16 %trunc0 to i32
  %ext1 = zext i16 %trunc1 to i32
  %ext2 = zext i16 %trunc2 to i32
  %ext3 = zext i16 %trunc3 to i32
  %ext4 = zext i16 %trunc4 to i32
  %ext5 = zext i16 %trunc5 to i32
  %ext6 = zext i16 %trunc6 to i32
  %ext7 = zext i16 %trunc7 to i32
  %ext8 = zext i16 %trunc8 to i32
  %ext9 = zext i16 %trunc9 to i32
  %ext10 = zext i16 %trunc10 to i32
  %ext11 = zext i16 %trunc11 to i32
  %ext12 = zext i16 %trunc12 to i32
  %ext13 = zext i16 %trunc13 to i32
  %ext14 = zext i16 %trunc14 to i32
  %ext15 = zext i16 %trunc15 to i32

  store volatile i32 %val0, i32 *%ptr
  store volatile i32 %val1, i32 *%ptr
  store volatile i32 %val2, i32 *%ptr
  store volatile i32 %val3, i32 *%ptr
  store volatile i32 %val4, i32 *%ptr
  store volatile i32 %val5, i32 *%ptr
  store volatile i32 %val6, i32 *%ptr
  store volatile i32 %val7, i32 *%ptr
  store volatile i32 %val8, i32 *%ptr
  store volatile i32 %val9, i32 *%ptr
  store volatile i32 %val10, i32 *%ptr
  store volatile i32 %val11, i32 *%ptr
  store volatile i32 %val12, i32 *%ptr
  store volatile i32 %val13, i32 *%ptr
  store volatile i32 %val14, i32 *%ptr
  store volatile i32 %val15, i32 *%ptr

  store volatile i32 %ext0, i32 *%ptr
  store volatile i32 %ext1, i32 *%ptr
  store volatile i32 %ext2, i32 *%ptr
  store volatile i32 %ext3, i32 *%ptr
  store volatile i32 %ext4, i32 *%ptr
  store volatile i32 %ext5, i32 *%ptr
  store volatile i32 %ext6, i32 *%ptr
  store volatile i32 %ext7, i32 *%ptr
  store volatile i32 %ext8, i32 *%ptr
  store volatile i32 %ext9, i32 *%ptr
  store volatile i32 %ext10, i32 *%ptr
  store volatile i32 %ext11, i32 *%ptr
  store volatile i32 %ext12, i32 *%ptr
  store volatile i32 %ext13, i32 *%ptr
  store volatile i32 %ext14, i32 *%ptr
  store volatile i32 %ext15, i32 *%ptr

  ret void
}
