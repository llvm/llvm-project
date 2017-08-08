; Check that unsized token types used by coroutine intrinsics do not cause
; assertion failures.
; RUN: opt < %s -msan -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*)
declare i1 @llvm.coro.alloc(token)

define void @foo() sanitize_memory {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %dyn.alloc.reqd = call i1  @llvm.coro.alloc(token %id)
  ret void
}

; CHECK: define void @foo
; CHECK-NEXT: entry:
; CHECK-NEXT: %id = call token @llvm.coro.id
; CHECK-NEXT: call i1 @llvm.coro.alloc(token %id)
; CHECK-NEXT: ret void
