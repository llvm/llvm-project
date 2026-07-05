; Test that memchr won't be converted to SRST if calls are
; marked with nobuiltin, eg. for sanitizers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare ptr@memchr(ptr %src, i16 %char, i32 %len)

; Test a simple forwarded call.
define ptr@f1(ptr %src, i16 %char, i32 %len) {
; CHECK-LABEL: f1:
; CHECK-NOT: srst
; CHECK: brasl %r14, memchr
; CHECK: br %r14
  %res = call ptr@memchr(ptr %src, i16 %char, i32 %len) nobuiltin
  ret ptr %res
}
