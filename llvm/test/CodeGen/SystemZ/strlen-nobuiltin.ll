; Test that strlen/strnlen won't be converted to SRST if calls are
; marked with nobuiltin, eg. for sanitizers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @strlen(i8 *%src)
declare i64 @strnlen(i8 *%src, i64 %len)

define i64 @f1(i32 %dummy, i8 *%src) {
; CHECK-LABEL: f1:
; CHECK: brasl %r14, strlen
; CHECK: br %r14
  %res = call i64 @strlen(i8 *%src) nobuiltin
  ret i64 %res
}

; Likewise for strnlen.
define i64 @f2(i64 %len, i8 *%src) {
; CHECK-LABEL: f2:
; CHECK-NOT: srst
; CHECK: brasl %r14, strnlen
; CHECK: br %r14
  %res = call i64 @strnlen(i8 *%src, i64 %len) nobuiltin
  ret i64 %res
}
