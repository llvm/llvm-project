; Check that we can handle the case when both alloc function and
; the user body consume the same argument.
; RUN: opt < %s -passes=coro-split -S | FileCheck %s

; using copy of this (as it would happen under -O0)
define i8* @f_copy(i64 %this_arg) "coroutine.presplit"="1" {
entry:
  %this.addr = alloca i64
  store i64 %this_arg, i64* %this.addr
  %this = load i64, i64* %this.addr
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @myAlloc(i64 %this, i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call void @print2(i64 %this)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl
}

; See if %this was added to the frame
; CHECK: %f_copy.Frame = type { void (%f_copy.Frame*)*, void (%f_copy.Frame*)*, i64, i1 }

; See that %this is spilled into the frame
; CHECK-LABEL: define i8* @f_copy(i64 %this_arg)
; CHECK:  %this.spill.addr = getelementptr inbounds %f_copy.Frame, %f_copy.Frame* %FramePtr, i32 0, i32 2
; CHECK:  store i64 %this_arg, i64* %this.spill.addr
; CHECK: ret i8* %hdl

; See that %this was loaded from the frame
; CHECK-LABEL: @f_copy.resume(
; CHECK:  %this.reload = load i64, i64* %this.reload.addr
; CHECK:  call void @print2(i64 %this.reload)
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

declare noalias i8* @myAlloc(i64, i32)
declare double @print(double)
declare void @print2(i64)
declare void @free(i8*)
