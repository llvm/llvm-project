; Check that we can handle spills of the result of the invoke instruction
; RUN: opt < %s -coro-split -S | FileCheck %s

define i8* @f(i64 %this) "coroutine.presplit"="1" personality i32 0 {
entry:
  %this.addr = alloca i64
  store i64 %this, i64* %this.addr
  %this1 = load i64, i64* %this.addr
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  %r = invoke double @print(double 0.0) to label %cont unwind label %pad

cont:
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call double @print(double %r)
  call void @print2(i64 %this1)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl
pad:
  %tok = cleanuppad within none []
  cleanupret from %tok unwind to caller
}

; See if the float was added to the frame
; CHECK-LABEL: %f.Frame = type { void (%f.Frame*)*, void (%f.Frame*)*, i1, i1, i64, double }

; See if the float was spilled into the frame
; CHECK-LABEL: @f(
; CHECK: %r = call double @print(
; CHECK: %r.spill.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 5
; CHECK: store double %r, double* %r.spill.addr
; CHECK: ret i8* %hdl

; See of the float was loaded from the frame
; CHECK-LABEL: @f.resume(
; CHECK: %r.reload = load double, double* %r.reload.addr
; CHECK: call double @print(double %r.reload)
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
declare void @print2(i64)
declare void @free(i8*)
