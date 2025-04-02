; RUN: opt -passes=gvn-hoist -S < %s | FileCheck %s

; CHECK: store
; CHECK-NOT: store

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

define void @f(ptr %p, i1 %arg) {
entry:
  switch i4 undef, label %if.then30 [
    i4 4, label %if.end
    i4 0, label %if.end
  ]

if.end:
  br label %if.end19

if.end19:
  br i1 %arg, label %e, label %e.thread

e.thread:
  store i8 0, ptr %p, align 4
  br label %if.then30

if.then30:
  call void @g()
  unreachable

e:
  store i8 0, ptr %p, align 4
  unreachable
}

declare void @g()
