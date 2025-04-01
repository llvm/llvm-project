; Test that store-load operation that crosses suspension point will not be eliminated by DSE before CoroSplit
; RUN: opt < %s -passes='dse,verify' -S | FileCheck %s

define void @fn(ptr align 8 %0) presplitcoroutine {
  %2 = alloca ptr, align 8
  %3 = alloca i8, align 1
  %4 = call token @llvm.coro.id(i32 16, ptr %2, ptr @fn, ptr null)
  %5 = call ptr @llvm.coro.begin(token %4, ptr null)
  %6 = call ptr @malloc(i64 1)
  call void @llvm.lifetime.start.p0(i64 8, ptr %2)
  store ptr %6, ptr %2, align 8
  %7 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.void(ptr %3, ptr %5, ptr @await_suspend_wrapper_void)
  %8 = call i8 @llvm.coro.suspend(token %7, i1 false)
  %9 = icmp ule i8 %8, 1
  br i1 %9, label %10, label %11

10:
  call void @llvm.lifetime.end.p0(i64 8, ptr %2)
  br label %11

11:
  %12 = call i1 @llvm.coro.end(ptr null, i1 false, token none)
  %13 = load ptr, ptr %2, align 8
  store ptr %13, ptr %0, align 8
; store when suspend, load when resume
; CHECK: store ptr null, ptr %2, align 8
  store ptr null, ptr %2, align 8
  ret void
}

declare ptr @malloc(i64)
declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.lifetime.start.p0(i64, ptr)
declare token @llvm.coro.save(ptr)
declare void @llvm.lifetime.end.p0(i64, ptr)
declare void @llvm.coro.await.suspend.void(ptr, ptr, ptr)
declare i8 @llvm.coro.suspend(token, i1)
declare i1 @llvm.coro.end(ptr, i1, token)
declare void @await_suspend_wrapper_void(ptr, ptr)
