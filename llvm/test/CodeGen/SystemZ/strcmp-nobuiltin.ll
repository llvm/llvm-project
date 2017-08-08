; Test that strcmp won't be converted to CLST if calls are
; marked with nobuiltin, eg. for sanitizers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare signext i32 @strcmp(i8 *%src1, i8 *%src2)

; Check a case where the result is used as an integer.
define i32 @f1(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f1:
; CHECK-NOT: clst
; CHECK: brasl %r14, strcmp
; CHECK: br %r14
  %res = call i32 @strcmp(i8 *%src1, i8 *%src2) nobuiltin
  ret i32 %res
}

; Check a case where the result is tested for equality.
define void @f2(i8 *%src1, i8 *%src2, i32 *%dest) {
; CHECK-LABEL: f2:
; CHECK-NOT: clst
; CHECK: brasl %r14, strcmp
; CHECK: br %r14
  %res = call i32 @strcmp(i8 *%src1, i8 *%src2) nobuiltin
  %cmp = icmp eq i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, i32 *%dest
  br label %exit

exit:
  ret void
}

; Test a case where the result is used both as an integer and for
; branching.
define i32 @f3(i8 *%src1, i8 *%src2, i32 *%dest) {
; CHECK-LABEL: f3:
; CHECK-NOT: clst
; CHECK: brasl %r14, strcmp
; CHECK: br %r14
entry:
  %res = call i32 @strcmp(i8 *%src1, i8 *%src2) nobuiltin
  %cmp = icmp slt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, i32 *%dest
  br label %exit

exit:
  ret i32 %res
}
