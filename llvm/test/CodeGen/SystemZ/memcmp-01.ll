; Test memcmp using CLC, with i32 results.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare signext i32 @memcmp(i8 *%src1, i8 *%src2, i64 %size)

; Zero-length comparisons should be optimized away.
define i32 @f1(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f1:
; CHECK: lhi %r2, 0
; CHECK: br %r14
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 0)
  ret i32 %res
}

; Check a case where the result is used as an integer.
define i32 @f2(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f2:
; CHECK: clc 0(2,%r2), 0(%r3)
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: rll %r2, [[REG]], 31
; CHECK: br %r14
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 2)
  ret i32 %res
}

; Check a case where the result is tested for equality.
define void @f3(i8 *%src1, i8 *%src2, i32 *%dest) {
; CHECK-LABEL: f3:
; CHECK: clc 0(3,%r2), 0(%r3)
; CHECK-NEXT: ber %r14
; CHECK: br %r14
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 3)
  %cmp = icmp eq i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, i32 *%dest
  br label %exit

exit:
  ret void
}

; Check a case where the result is tested for inequality.
define void @f4(i8 *%src1, i8 *%src2, i32 *%dest) {
; CHECK-LABEL: f4:
; CHECK: clc 0(4,%r2), 0(%r3)
; CHECK-NEXT: blhr %r14
; CHECK: br %r14
entry:
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 4)
  %cmp = icmp ne i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, i32 *%dest
  br label %exit

exit:
  ret void
}

; Check a case where the result is tested via slt.
define void @f5(i8 *%src1, i8 *%src2, i32 *%dest) {
; CHECK-LABEL: f5:
; CHECK: clc 0(5,%r2), 0(%r3)
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 5)
  %cmp = icmp slt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, i32 *%dest
  br label %exit

exit:
  ret void
}

; Check a case where the result is tested for sgt.
define void @f6(i8 *%src1, i8 *%src2, i32 *%dest) {
; CHECK-LABEL: f6:
; CHECK: clc 0(6,%r2), 0(%r3)
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 6)
  %cmp = icmp sgt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, i32 *%dest
  br label %exit

exit:
  ret void
}

; Check the upper end of the CLC range.  Here the result is used both as
; an integer and for branching.
define i32 @f7(i8 *%src1, i8 *%src2, i32 *%dest) {
; CHECK-LABEL: f7:
; CHECK: clc 0(256,%r2), 0(%r3)
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: rll %r2, [[REG]], 31
; CHECK: blr %r14
; CHECK: br %r14
entry:
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 256)
  %cmp = icmp slt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, i32 *%dest
  br label %exit

exit:
  ret i32 %res
}

; 257 bytes needs two CLCs.
define i32 @f8(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f8:
; CHECK: clc 0(256,%r2), 0(%r3)
; CHECK: jlh [[LABEL:\..*]]
; CHECK: clc 256(1,%r2), 256(%r3)
; CHECK: [[LABEL]]:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: br %r14
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 257)
  ret i32 %res
}

; Test a comparison of 258 bytes in which the CC result can be used directly.
define void @f9(i8 *%src1, i8 *%src2, i32 *%dest) {
; CHECK-LABEL: f9:
; CHECK: clc 0(256,%r2), 0(%r3)
; CHECK: jlh [[LABEL:\..*]]
; CHECK: clc 256(1,%r2), 256(%r3)
; CHECK: [[LABEL]]:
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 257)
  %cmp = icmp slt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, i32 *%dest
  br label %exit

exit:
  ret void
}

; Test the largest size that can use two CLCs.
define i32 @f10(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f10:
; CHECK: clc 0(256,%r2), 0(%r3)
; CHECK: jlh [[LABEL:\..*]]
; CHECK: clc 256(256,%r2), 256(%r3)
; CHECK: [[LABEL]]:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: br %r14
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 512)
  ret i32 %res
}

; Test the smallest size that needs 3 CLCs.
define i32 @f11(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f11:
; CHECK: clc 0(256,%r2), 0(%r3)
; CHECK: jlh [[LABEL:\..*]]
; CHECK: clc 256(256,%r2), 256(%r3)
; CHECK: jlh [[LABEL]]
; CHECK: clc 512(1,%r2), 512(%r3)
; CHECK: [[LABEL]]:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: br %r14
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 513)
  ret i32 %res
}

; Test the largest size than can use 3 CLCs.
define i32 @f12(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f12:
; CHECK: clc 0(256,%r2), 0(%r3)
; CHECK: jlh [[LABEL:\..*]]
; CHECK: clc 256(256,%r2), 256(%r3)
; CHECK: jlh [[LABEL]]
; CHECK: clc 512(256,%r2), 512(%r3)
; CHECK: [[LABEL]]:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: br %r14
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 768)
  ret i32 %res
}

; The next size up uses a loop instead.  We leave the more complicated
; loop tests to memcpy-01.ll, which shares the same form.
define i32 @f13(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f13:
; CHECK: lghi [[COUNT:%r[0-5]]], 3
; CHECK: [[LOOP:.L[^:]*]]:
; CHECK: clc 0(256,%r2), 0(%r3)
; CHECK: jlh [[LABEL:\..*]]
; CHECK-DAG: la %r2, 256(%r2)
; CHECK-DAG: la %r3, 256(%r3)
; CHECK: brctg [[COUNT]], [[LOOP]]
; CHECK: clc 0(1,%r2), 0(%r3)
; CHECK: [[LABEL]]:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: br %r14
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 769)
  ret i32 %res
}
