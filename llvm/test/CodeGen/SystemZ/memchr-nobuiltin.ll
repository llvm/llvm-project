; Test that memchr won't be converted to SRST if calls are
; marked with nobuiltin, eg. for sanitizers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i8 *@memchr(i8 *%src, i16 %char, i32 %len)

; Test a simple forwarded call.
define i8 *@f1(i8 *%src, i16 %char, i32 %len) {
; CHECK-LABEL: f1:
; CHECK-NOT: srst
; CHECK: brasl %r14, memchr
; CHECK: br %r14
  %res = call i8 *@memchr(i8 *%src, i16 %char, i32 %len) nobuiltin
  ret i8 *%res
}
