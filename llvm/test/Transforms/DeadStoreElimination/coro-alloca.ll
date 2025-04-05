; Test that store-load operation that crosses suspension point will not be eliminated by DSE before CoroSplit
; RUN: opt < %s -passes='dse' -S | FileCheck %s

define void @fn(ptr align 8 %arg) presplitcoroutine {
  %promise = alloca ptr, align 8
  %awaiter = alloca i8, align 1
  %id = call token @llvm.coro.id(i32 16, ptr %promise, ptr @fn, ptr null)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %mem = call ptr @malloc(i64 1)
  call void @llvm.lifetime.start.p0(i64 8, ptr %promise)
  store ptr %mem, ptr %promise, align 8
  %save = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.void(ptr %awaiter, ptr %hdl, ptr @await_suspend_wrapper_void)
  %sp = call i8 @llvm.coro.suspend(token %save, i1 false)
  %flag = icmp ule i8 %sp, 1
  br i1 %flag, label %resume, label %suspend

resume:
  call void @llvm.lifetime.end.p0(i64 8, ptr %promise)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(ptr null, i1 false, token none)
  %temp = load ptr, ptr %promise, align 8
  store ptr %temp, ptr %arg, align 8
; store when suspend, load when resume
; CHECK: store ptr null, ptr %promise, align 8
  store ptr null, ptr %promise, align 8
  ret void
}

declare ptr @malloc(i64)
declare void @await_suspend_wrapper_void(ptr, ptr)
