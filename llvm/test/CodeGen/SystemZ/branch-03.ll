; Test all condition-code masks that are relevant for unsigned integer
; comparisons.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
define void @f1(i32 *%src, i32 %target) {
; CHECK-LABEL: f1:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: cl %r3, 0(%r2)
; CHECK-NEXT: jle .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32, i32 *%src
  %cond = icmp ule i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f2(i32 *%src, i32 %target) {
; CHECK-LABEL: f2:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: cl %r3, 0(%r2)
; CHECK-NEXT: jl .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32, i32 *%src
  %cond = icmp ult i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f3(i32 *%src, i32 %target) {
; CHECK-LABEL: f3:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: cl %r3, 0(%r2)
; CHECK-NEXT: jh .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32, i32 *%src
  %cond = icmp ugt i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f4(i32 *%src, i32 %target) {
; CHECK-LABEL: f4:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: cl %r3, 0(%r2)
; CHECK-NEXT: jhe .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32, i32 *%src
  %cond = icmp uge i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
