; Test all condition-code masks that are relevant for signed integer
; comparisons, in cases where a separate branch is better than COMPARE
; AND BRANCH.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(i32 *%src, i32 %target) {
; CHECK-LABEL: f1:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: c %r3, 0(%r2)
; CHECK-NEXT: je .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32, i32 *%src
  %cond = icmp eq i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f2(i32 *%src, i32 %target) {
; CHECK-LABEL: f2:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: c %r3, 0(%r2)
; CHECK-NEXT: jlh .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32, i32 *%src
  %cond = icmp ne i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f3(i32 *%src, i32 %target) {
; CHECK-LABEL: f3:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: c %r3, 0(%r2)
; CHECK-NEXT: jle .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32, i32 *%src
  %cond = icmp sle i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f4(i32 *%src, i32 %target) {
; CHECK-LABEL: f4:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: c %r3, 0(%r2)
; CHECK-NEXT: jl .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32, i32 *%src
  %cond = icmp slt i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f5(i32 *%src, i32 %target) {
; CHECK-LABEL: f5:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: c %r3, 0(%r2)
; CHECK-NEXT: jh .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32, i32 *%src
  %cond = icmp sgt i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f6(i32 *%src, i32 %target) {
; CHECK-LABEL: f6:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: c %r3, 0(%r2)
; CHECK-NEXT: jhe .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32, i32 *%src
  %cond = icmp sge i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
