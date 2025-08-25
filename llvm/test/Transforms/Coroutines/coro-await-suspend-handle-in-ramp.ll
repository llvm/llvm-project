; Tests lowerings of different versions of coro.await.suspend
; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split),simplifycfg' -S | FileCheck %s

%Awaiter = type {}

define void @f() presplitcoroutine {
entry:
  %awaiter = alloca %Awaiter
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  call void @llvm.coro.await.suspend.handle(ptr %awaiter, ptr %hdl, ptr @await_suspend_wrapper_handle)
  %suspend.init = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %suspend.init, label %ret [
    i8 0, label %step
    i8 1, label %cleanup
  ]

; Check the calling convention for resuming function is fastcc
; CHECK:     define {{[^@]*}} @f()
; CHECK:      entry:
; CHECK:        %[[NEXT_HDL:.+]] = call ptr @await_suspend_wrapper_handle(
; CHECK-NEXT:   %[[CONT:.+]] = call ptr @llvm.coro.subfn.addr(ptr %[[NEXT_HDL]], i8 0)
; CHECK-NEXT:   call fastcc void %[[CONT]](ptr %[[NEXT_HDL]])
step:
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %ret

ret:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret void
}

; check that we were haven't accidentally went out of @f body
; CHECK-LABEL: @f.resume(
; CHECK-LABEL: @f.destroy(
; CHECK-LABEL: @f.cleanup(

declare ptr @await_suspend_wrapper_handle(ptr, ptr)

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.await.suspend.handle(ptr, ptr, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare noalias ptr @malloc(i32)
declare void @free(ptr)
