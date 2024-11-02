; Tests lowerings of different versions of coro.await.suspend
; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split),simplifycfg' -S | FileCheck %s

%Awaiter = type {}

; CHECK:     define {{[^@]*}} @f.resume(ptr {{[^%]*}} %[[HDL:.+]])
; CHECK:       %[[AWAITER:.+]] = getelementptr inbounds %f.Frame, ptr %[[HDL]], i32 0, i32 0
define void @f() presplitcoroutine {
entry:
  %awaiter = alloca %Awaiter
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %suspend.init = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %suspend.init, label %ret [
    i8 0, label %step
    i8 1, label %cleanup
  ]

; CHECK:        call void @await_suspend_wrapper_void(ptr %[[AWAITER]], ptr %[[HDL]])
; CHECK-NEXT:   br label %{{.*}}
step:
  %save = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.void(ptr %awaiter, ptr %hdl, ptr @await_suspend_wrapper_void)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %ret [
    i8 0, label %step1
    i8 1, label %cleanup
  ]

; CHECK:        %[[RESUME:.+]] = call i1 @await_suspend_wrapper_bool(ptr %[[AWAITER]], ptr %[[HDL]])
; CHECK-NEXT:   br i1 %[[RESUME]], label %{{[^,]+}}, label %[[STEP2:.+]]
step1:
  %save1 = call token @llvm.coro.save(ptr null)
  %resume.bool = call i1 @llvm.coro.await.suspend.bool(ptr %awaiter, ptr %hdl, ptr @await_suspend_wrapper_bool)
  br i1 %resume.bool, label %suspend.cond, label %step2

suspend.cond:
  %suspend1 = call i8 @llvm.coro.suspend(token %save1, i1 false)
  switch i8 %suspend1, label %ret [
    i8 0, label %step2
    i8 1, label %cleanup
  ]

; CHECK:      [[STEP2]]:
; CHECK:        %[[NEXT_HDL:.+]] = call ptr @await_suspend_wrapper_handle(ptr %[[AWAITER]], ptr %[[HDL]])
; CHECK-NEXT:   %[[CONT:.+]] = call ptr @llvm.coro.subfn.addr(ptr %[[NEXT_HDL]], i8 0)
; CHECK-NEXT:   musttail call {{.*}} void %[[CONT]](ptr %[[NEXT_HDL]])
step2:
  %save2 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.handle(ptr %awaiter, ptr %hdl, ptr @await_suspend_wrapper_handle)
  %suspend2 = call i8 @llvm.coro.suspend(token %save2, i1 false)
  switch i8 %suspend2, label %ret [
    i8 0, label %step3
    i8 1, label %cleanup
  ]

step3:
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %ret

ret:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret void
}

; check that we were haven't accidentally went out of @f.resume body
; CHECK-LABEL: @f.destroy(
; CHECK-LABEL: @f.cleanup(

declare void @await_suspend_wrapper_void(ptr, ptr)
declare i1 @await_suspend_wrapper_bool(ptr, ptr)
declare ptr @await_suspend_wrapper_handle(ptr, ptr)

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.await.suspend.void(ptr, ptr, ptr)
declare i1 @llvm.coro.await.suspend.bool(ptr, ptr, ptr)
declare void @llvm.coro.await.suspend.handle(ptr, ptr, ptr)
declare i1 @llvm.coro.end(ptr, i1, token)

declare noalias ptr @malloc(i32)
declare void @free(ptr)
