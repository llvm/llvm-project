; Test zero extensions from an i32 to an i64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test register extension, starting with an i32.
define i64 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: llgfr %r2, %r2
; CHECK: br %r14
  %ext = zext i32 %a to i64
  ret i64 %ext
}

; ...and again with an i64.
define i64 @f2(i64 %a) {
; CHECK-LABEL: f2:
; CHECK: llgfr %r2, %r2
; CHECK: br %r14
  %word = trunc i64 %a to i32
  %ext = zext i32 %word to i64
  ret i64 %ext
}

; Check ANDs that are equivalent to zero extension.
define i64 @f3(i64 %a) {
; CHECK-LABEL: f3:
; CHECK: llgfr %r2, %r2
; CHECK: br %r14
  %ext = and i64 %a, 4294967295
  ret i64 %ext
}

; Check LLGF with no displacement.
define i64 @f4(i32 *%src) {
; CHECK-LABEL: f4:
; CHECK: llgf %r2, 0(%r2)
; CHECK: br %r14
  %word = load i32, i32 *%src
  %ext = zext i32 %word to i64
  ret i64 %ext
}

; Check the high end of the LLGF range.
define i64 @f5(i32 *%src) {
; CHECK-LABEL: f5:
; CHECK: llgf %r2, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131071
  %word = load i32, i32 *%ptr
  %ext = zext i32 %word to i64
  ret i64 %ext
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f6(i32 *%src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, 524288
; CHECK: llgf %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131072
  %word = load i32, i32 *%ptr
  %ext = zext i32 %word to i64
  ret i64 %ext
}

; Check the high end of the negative LLGF range.
define i64 @f7(i32 *%src) {
; CHECK-LABEL: f7:
; CHECK: llgf %r2, -4(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -1
  %word = load i32, i32 *%ptr
  %ext = zext i32 %word to i64
  ret i64 %ext
}

; Check the low end of the LLGF range.
define i64 @f8(i32 *%src) {
; CHECK-LABEL: f8:
; CHECK: llgf %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131072
  %word = load i32, i32 *%ptr
  %ext = zext i32 %word to i64
  ret i64 %ext
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f9(i32 *%src) {
; CHECK-LABEL: f9:
; CHECK: agfi %r2, -524292
; CHECK: llgf %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131073
  %word = load i32, i32 *%ptr
  %ext = zext i32 %word to i64
  ret i64 %ext
}

; Check that LLGF allows an index.
define i64 @f10(i64 %src, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: llgf %r2, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i32 *
  %word = load i32, i32 *%ptr
  %ext = zext i32 %word to i64
  ret i64 %ext
}
