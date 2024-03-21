; RUN: llc < %s | FileCheck %s --check-prefix=CHECK-SMALL
; RUN: llc --code-model=medium -large-data-threshold=100 < %s | FileCheck %s --check-prefix=CHECK-SMALL
; RUN: llc --code-model=medium < %s | FileCheck %s --check-prefix=CHECK-LARGE
; RUN: llc --code-model=large -large-data-threshold=100 < %s | FileCheck %s --check-prefix=CHECK-SMALL
; RUN: llc --code-model=large < %s | FileCheck %s --check-prefix=CHECK-LARGE
; RUN: llc --code-model=kernel < %s | FileCheck %s --check-prefix=CHECK-SMALL

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external dso_local global i32, align 4

declare void @f()

define void @foo(i64 %b) {
; CHECK-LARGE: cmpq %rax, %rdi
; CHECK-SMALL: cmpq $a, %rdi
entry:
  %cmp = icmp eq i64 %b, ptrtoint (ptr @a to i64)
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @f()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

