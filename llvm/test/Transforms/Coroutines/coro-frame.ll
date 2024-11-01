; Check that we can handle spills of the result of the invoke instruction
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @f(i64 %this) presplitcoroutine personality i32 0 {
entry:
  %this.addr = alloca i64
  store i64 %this, ptr %this.addr
  %this1 = load i64, ptr %this.addr
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
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
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0)
  ret ptr %hdl
pad:
  %tok = cleanuppad within none []
  cleanupret from %tok unwind to caller
}

; See if the float was added to the frame
; CHECK-LABEL: %f.Frame = type { ptr, ptr, double, i64, i1 }

; See if the float was spilled into the frame
; CHECK-LABEL: @f(
; CHECK: %r = call double @print(
; CHECK: %r.spill.addr = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 2
; CHECK: store double %r, ptr %r.spill.addr
; CHECK: ret ptr %hdl

; See if the float was loaded from the frame
; CHECK-LABEL: @f.resume(ptr noundef nonnull align 8
; CHECK: %r.reload = load double, ptr %r.reload.addr
; CHECK: call double @print(double %r.reload)
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
declare void @print2(i64)
declare void @free(ptr)
