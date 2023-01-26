; Test that functions with dynamic allocas get inlined in a case where
; naively inlining it would result in a miscompilation.
; Functions with dynamic allocas can only be inlined into functions that
; already have dynamic allocas.

; RUN: opt < %s -passes=inline -S | FileCheck %s
;
; FIXME: This test is xfailed because the inline cost rewrite disabled *all*
; inlining of functions which contain a dynamic alloca. It should be re-enabled
; once that functionality is restored.
; XFAIL: *

declare void @ext(ptr)

define internal void @callee(i32 %N) {
  %P = alloca i32, i32 %N
  call void @ext(ptr %P)
  ret void
}

define void @foo(i32 %N) {
; CHECK-LABEL: @foo(
; CHECK: alloca i32, i32 %{{.*}}
; CHECK: call ptr @llvm.stacksave()
; CHECK: alloca i32, i32 %{{.*}}
; CHECK: call void @ext
; CHECK: call void @llvm.stackrestore
; CHECK: ret

entry:
  %P = alloca i32, i32 %N
  call void @ext(ptr %P)
  br label %loop

loop:
  %count = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %count, 1
  call void @callee(i32 %N)
  %cond = icmp eq i32 %count, 100000
  br i1 %cond, label %out, label %loop

out:
  ret void
}

