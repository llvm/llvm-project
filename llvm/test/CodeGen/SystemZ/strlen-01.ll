; Test strlen using SRST, i64 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @strlen(i8 *%src)
declare i64 @strnlen(i8 *%src, i64 %len)

; Test strlen with its proper i64 prototype.  It would also be valid for
; the uses of %r3 and REG after the LGR to be swapped.
define i64 @f1(i32 %dummy, i8 *%src) {
; CHECK-LABEL: f1:
; CHECK-DAG: lhi %r0, 0
; CHECK-DAG: lghi %r2, 0
; CHECK-DAG: lgr [[REG:%r[145]]], %r3
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK-NEXT: srst %r2, [[REG]]
; CHECK-NEXT: jo [[LABEL]]
; CHECK-NEXT: %bb.{{[0-9]+}}
; CHECK-NEXT: sgr %r2, %r3
; CHECK: br %r14
  %res = call i64 @strlen(i8 *%src)
  ret i64 %res
}

; Test strnlen with its proper i64 prototype.
define i64 @f2(i64 %len, i8 *%src) {
; CHECK-LABEL: f2:
; CHECK-DAG: agr %r2, %r3
; CHECK-DAG: lhi %r0, 0
; CHECK-DAG: lgr [[REG:%r[145]]], %r3
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK-NEXT: srst %r2, [[REG]]
; CHECK-NEXT: jo [[LABEL]]
; CHECK-NEXT: %bb.{{[0-9]+}}
; CHECK-NEXT: sgr %r2, %r3
; CHECK: br %r14
  %res = call i64 @strnlen(i8 *%src, i64 %len)
  ret i64 %res
}
