; RUN: opt < %s -passes='module-inline' -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define ptr @foo() {
; CHECK: define ptr @foo()
; CHECK: %{{.*}} = call ptr @bar(), !inline_history ![[HIST:[0-9]+]]
entry:
  %0 = call ptr @bar()
  ret ptr %0
}

; Function Attrs: minsize
define ptr @bar() #0 {
entry:
  %0 = load i64, ptr null, align 8
  call void @baz()
  ret ptr null
}

; Function Attrs: minsize
define void @baz() #0 {
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %0 = call ptr @bar()
  br label %loop
}

attributes #0 = { minsize }

; CHECK: ![[HIST]] = !{ptr @{{baz|bar}}, ptr @{{baz|bar}}}
