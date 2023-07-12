; Check that we would take care of the value written to promise before @llvm.coro.begin.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

%"class.task::promise_type" = type { [64 x i8] }

declare void @consume(i32*)
declare void @consume2(%"class.task::promise_type"*)

define ptr @f() presplitcoroutine {
entry:
  %data = alloca i32, align 4
  %__promise = alloca %"class.task::promise_type", align 64
  %id = call token @llvm.coro.id(i32 0, ptr %__promise, ptr null, ptr null)
  call void @consume2(%"class.task::promise_type"* %__promise)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  call void @consume(i32* %data)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call void @consume(i32* %data)
  call void @consume2(%"class.task::promise_type"* %__promise)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0)
  ret ptr %hdl
}

; CHECK-LABEL: %f.Frame = type { ptr, ptr, i32, i1, [43 x i8], %"class.task::promise_type" }

; CHECK-LABEL: @f(
; CHECK: %__promise = alloca %"class.task::promise_type"
; CHECK: call void @consume2(ptr %__promise)
; CHECK: call{{.*}}@llvm.coro.begin
; CHECK: %[[PROMISE_ADDR:.+]] = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 5
; CHECK: %[[PROMISE_VALUE:.+]] = load %"class.task::promise_type", ptr %__promise
; CHECK: store %"class.task::promise_type" %[[PROMISE_VALUE]], ptr %[[PROMISE_ADDR]]

; CHECK-LABEL: @f.resume(
; CHECK: %[[DATA:.+]] = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 5
; CHECK: call void @consume2(ptr %[[DATA]])
; CHECK: ret void

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1)

declare noalias ptr @malloc(i32)
declare double @print(double)
declare void @free(ptr)
