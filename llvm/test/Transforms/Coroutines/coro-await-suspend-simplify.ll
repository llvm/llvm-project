; Tests that CoroSplit can succesfully see through await suspend wrapper
; and remove redundant suspend points only if there are no calls in the wrapper
; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split),simplifycfg' -S | FileCheck %s

; CHECK-LABEL: @f(
; CHECK:        call ptr @await_suspend_wrapper(
; CHECK-NEXT:   call void @free(
define void @f() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %save = call token @llvm.coro.save(ptr null)
  %hdl.resume = call ptr @llvm.coro.await.suspend.handle(ptr null, ptr %hdl, ptr @await_suspend_wrapper)
  call void @llvm.coro.resume(ptr %hdl.resume)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %ret [
    i8 0, label %resume
    i8 1, label %cleanup
  ]

resume:
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %ret

ret:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret void
}

define ptr @await_suspend_wrapper(ptr, ptr %hdl) {
  ret ptr %hdl
}

; CHECK-LABEL: @f1(
; CHECK:        %[[HDL:.+]] = call ptr @await_suspend_wrapper1(
; CHECK-NEXT:   %[[FN:.+]] = call ptr @llvm.coro.subfn.addr(ptr %[[HDL]], i8 0)
; CHECK-NEXT:   call {{.*}} %[[FN]](ptr %[[HDL]])
define void @f1() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %save = call token @llvm.coro.save(ptr null)
  %hdl.resume = call ptr @llvm.coro.await.suspend.handle(ptr null, ptr %hdl, ptr @await_suspend_wrapper1)
  call void @llvm.coro.resume(ptr %hdl.resume)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %ret [
    i8 0, label %resume
    i8 1, label %cleanup
  ]

resume:
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %ret

ret:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret void
}

define ptr @await_suspend_wrapper1(ptr, ptr %hdl) {
  call void @external()
  ret ptr %hdl
}

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare ptr @llvm.coro.await.suspend.handle(ptr, ptr, ptr)
declare i1 @llvm.coro.end(ptr, i1, token)

declare void @external()

declare noalias ptr @malloc(i32)
declare void @free(ptr)
