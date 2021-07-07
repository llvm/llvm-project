; Check that promise object is reloaded from the correct index of the coro frame.
; RUN: opt < %s -passes='cgscc(coro-split),simplify-cfg,early-cse' -S | FileCheck %s

%"class.task::promise_type" = type { [64 x i8] }

declare void @consume(i32*)
declare void @consume2(%"class.task::promise_type"*)

define i8* @f() "coroutine.presplit"="1" {
entry:
  %data = alloca i32, align 4
  %__promise = alloca %"class.task::promise_type", align 64
  %pv = bitcast %"class.task::promise_type"* %__promise to i8*
  %id = call token @llvm.coro.id(i32 0, i8* %pv, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  call void @consume(i32* %data)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call void @consume(i32* %data)
  call void @consume2(%"class.task::promise_type"* %__promise)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl
}

; CHECK-LABEL: %f.Frame = type { void (%f.Frame*)*, void (%f.Frame*)*, i32, i1, [43 x i8], %"class.task::promise_type" }

; CHECK-LABEL: @f.resume(
; CHECK: %[[DATA:.+]] = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 5
; CHECK: call void @consume2(%"class.task::promise_type"* %[[DATA]])
; CHECK: ret void

declare i8* @llvm.coro.free(token, i8*)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i1 @llvm.coro.alloc(token)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)

declare noalias i8* @malloc(i32)
declare double @print(double)
declare void @free(i8*)
