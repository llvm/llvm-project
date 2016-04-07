; Test memchr using SRST, with a weird but usable prototype.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -verify-machineinstrs | FileCheck %s

declare i8 *@memchr(i8 *%src, i16 %char, i32 %len)

; Test a simple forwarded call.
define i8 *@f1(i8 *%src, i16 %char, i32 %len) {
; CHECK-LABEL: f1:
; CHECK-DAG: lgr [[REG:%r[1-5]]], %r2
; CHECK-DAG: algfr %r2, %r4
; CHECK-DAG: llcr %r0, %r3
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: srst %r2, [[REG]]
; CHECK-NEXT: jo [[LABEL]]
; CHECK: blr %r14
; CHECK: lghi %r2, 0
; CHECK: br %r14
  %res = call i8 *@memchr(i8 *%src, i16 %char, i32 %len)
  ret i8 *%res
}
