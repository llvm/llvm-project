; Test memcmp using CLC, with i64 results.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @memcmp(i8 *%src1, i8 *%src2, i64 %size)

; Zero-length comparisons should be optimized away.
define i64 @f1(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f1:
; CHECK: lghi %r2, 0
; CHECK: br %r14
  %res = call i64 @memcmp(i8 *%src1, i8 *%src2, i64 0)
  ret i64 %res
}

; Check a case where the result is used as an integer.
define i64 @f2(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f2:
; CHECK: clc 0(2,%r2), 0(%r3)
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: rll [[REG]], [[REG]], 31
; CHECK: lgfr %r2, [[REG]]
; CHECK: br %r14
  %res = call i64 @memcmp(i8 *%src1, i8 *%src2, i64 2)
  ret i64 %res
}

; Check a case where the result is tested for equality.
define void @f3(i8 *%src1, i8 *%src2, i64 *%dest) {
; CHECK-LABEL: f3:
; CHECK: clc 0(3,%r2), 0(%r3)
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %res = call i64 @memcmp(i8 *%src1, i8 *%src2, i64 3)
  %cmp = icmp eq i64 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 0, i64 *%dest
  br label %exit

exit:
  ret void
}

; Check a case where the result is tested for inequality.
define void @f4(i8 *%src1, i8 *%src2, i64 *%dest) {
; CHECK-LABEL: f4:
; CHECK: clc 0(4,%r2), 0(%r3)
; CHECK-NEXT: blhr %r14
; CHECK: br %r14
entry:
  %res = call i64 @memcmp(i8 *%src1, i8 *%src2, i64 4)
  %cmp = icmp ne i64 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 0, i64 *%dest
  br label %exit

exit:
  ret void
}

; Check a case where the result is tested via slt.
define void @f5(i8 *%src1, i8 *%src2, i64 *%dest) {
; CHECK-LABEL: f5:
; CHECK: clc 0(5,%r2), 0(%r3)
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %res = call i64 @memcmp(i8 *%src1, i8 *%src2, i64 5)
  %cmp = icmp slt i64 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 0, i64 *%dest
  br label %exit

exit:
  ret void
}

; Check a case where the result is tested for sgt.
define void @f6(i8 *%src1, i8 *%src2, i64 *%dest) {
; CHECK-LABEL: f6:
; CHECK: clc 0(6,%r2), 0(%r3)
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %res = call i64 @memcmp(i8 *%src1, i8 *%src2, i64 6)
  %cmp = icmp sgt i64 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 0, i64 *%dest
  br label %exit

exit:
  ret void
}

; Check the upper end of the CLC range.  Here the result is used both as
; an integer and for branching.
define i64 @f7(i8 *%src1, i8 *%src2, i64 *%dest) {
; CHECK-LABEL: f7:
; CHECK: clc 0(256,%r2), 0(%r3)
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: rll [[REG]], [[REG]], 31
; CHECK: lgfr %r2, [[REG]]
; CHECK: blr %r14
; CHECK: br %r14
entry:
  %res = call i64 @memcmp(i8 *%src1, i8 *%src2, i64 256)
  %cmp = icmp slt i64 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i64 0, i64 *%dest
  br label %exit

exit:
  ret i64 %res
}

; 257 bytes needs two CLCs.
define i64 @f8(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f8:
; CHECK: clc 0(256,%r2), 0(%r3)
; CHECK: jlh [[LABEL:\..*]]
; CHECK: clc 256(1,%r2), 256(%r3)
; CHECK: [[LABEL]]:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: br %r14
  %res = call i64 @memcmp(i8 *%src1, i8 *%src2, i64 257)
  ret i64 %res
}
