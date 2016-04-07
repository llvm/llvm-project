; Test strcmp using CLST, i64 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @strcmp(i8 *%src1, i8 *%src2)

; Check a case where the result is used as an integer.
define i64 @f1(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f1:
; CHECK: lhi %r0, 0
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: clst %r2, %r3
; CHECK-NEXT: jo [[LABEL]]
; CHECK-NEXT: BB#{{[0-9]+}}
; CHECK-NEXT: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: rll [[REG]], [[REG]], 31
; CHECK: lgfr %r2, [[REG]]
; CHECK: br %r14
  %res = call i64 @strcmp(i8 *%src1, i8 *%src2)
  ret i64 %res
}

; Check a case where the result is tested for equality.
define void @f2(i8 *%src1, i8 *%src2, i64 *%dest) {
; CHECK-LABEL: f2:
; CHECK: lhi %r0, 0
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: clst %r2, %r3
; CHECK-NEXT: jo [[LABEL]]
; CHECK-NEXT: BB#{{[0-9]+}}
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %res = call i64 @strcmp(i8 *%src1, i8 *%src2)
  %cmp = icmp eq i64 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 0, i64 *%dest
  br label %exit

exit:
  ret void
}

; Test a case where the result is used both as an integer and for
; branching.
define i64 @f3(i8 *%src1, i8 *%src2, i64 *%dest) {
; CHECK-LABEL: f3:
; CHECK: lhi %r0, 0
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: clst %r2, %r3
; CHECK-NEXT: jo [[LABEL]]
; CHECK-NEXT: BB#{{[0-9]+}}
; CHECK-NEXT: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: rll [[REG]], [[REG]], 31
; CHECK: lgfr %r2, [[REG]]
; CHECK: blr %r14
; CHECK: br %r14
entry:
  %res = call i64 @strcmp(i8 *%src1, i8 *%src2)
  %cmp = icmp slt i64 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 0, i64 *%dest
  br label %exit

exit:
  ret i64 %res
}
