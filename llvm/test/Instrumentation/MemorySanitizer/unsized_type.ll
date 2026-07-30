; Check that unsized token types used by coroutine intrinsics do not cause
; assertion failures.
; RUN: opt < %s -S 2>&1 -passes=msan | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr)
declare i1 @llvm.coro.alloc(token)

define void @foo() sanitize_memory {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %dyn.alloc.reqd = call i1  @llvm.coro.alloc(token %id)
  ret void
}

; CHECK: define void @foo
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.donothing
; CHECK-NEXT: %id = call token @llvm.coro.id
; CHECK-NEXT: call i1 @llvm.coro.alloc(token %id)
; CHECK-NEXT: ret void
