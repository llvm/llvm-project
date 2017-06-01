; Test 16-bit byteswaps from memory to registers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i16 @llvm.bswap.i16(i16 %a)

; Check LRVH with no displacement.
define i16 @f1(i16 *%src) {
; CHECK-LABEL: f1:
; CHECK: lrvh %r2, 0(%r2)
; CHECK: br %r14
  %a = load i16 , i16 *%src
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  ret i16 %swapped
}

; Check the high end of the aligned LRVH range.
define i16 @f2(i16 *%src) {
; CHECK-LABEL: f2:
; CHECK: lrvh %r2, 524286(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 262143
  %a = load i16 , i16 *%ptr
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  ret i16 %swapped
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i16 @f3(i16 *%src) {
; CHECK-LABEL: f3:
; CHECK: agfi %r2, 524288
; CHECK: lrvh %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 262144
  %a = load i16 , i16 *%ptr
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  ret i16 %swapped
}

; Check the high end of the negative aligned LRVH range.
define i16 @f4(i16 *%src) {
; CHECK-LABEL: f4:
; CHECK: lrvh %r2, -2(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -1
  %a = load i16 , i16 *%ptr
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  ret i16 %swapped
}

; Check the low end of the LRVH range.
define i16 @f5(i16 *%src) {
; CHECK-LABEL: f5:
; CHECK: lrvh %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -262144
  %a = load i16 , i16 *%ptr
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  ret i16 %swapped
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i16 @f6(i16 *%src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, -524290
; CHECK: lrvh %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%src, i64 -262145
  %a = load i16 , i16 *%ptr
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  ret i16 %swapped
}

; Check that LRVH allows an index.
define i16 @f7(i64 %src, i64 %index) {
; CHECK-LABEL: f7:
; CHECK: lrvh %r2, 524287({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i16 *
  %a = load i16 , i16 *%ptr
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  ret i16 %swapped
}

; Check that volatile accesses do not use LRVH, which might access the
; storage multple times.
define i16 @f8(i16 *%src) {
; CHECK-LABEL: f8:
; CHECK: lh [[REG:%r[0-5]]], 0(%r2)
; CHECK: lrvr %r2, [[REG]]
; CHECK: br %r14
  %a = load volatile i16 , i16 *%src
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  ret i16 %swapped
}
