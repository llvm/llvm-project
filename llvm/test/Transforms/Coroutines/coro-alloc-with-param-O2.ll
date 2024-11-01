; Check that we can handle the case when both alloc function and
; the user body consume the same argument.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

; using this directly (as it would happen under -O2)
define ptr @f_direct(i64 %this) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @myAlloc(i64 %this, i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call void @print2(i64 %this)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0)
  ret ptr %hdl
}

; See if %this was added to the frame
; CHECK: %f_direct.Frame = type { ptr, ptr, i64, i1 }

; See that %this is spilled into the frame
; CHECK-LABEL: define ptr @f_direct(i64 %this)
; CHECK: %this.spill.addr = getelementptr inbounds %f_direct.Frame, ptr %hdl, i32 0, i32 2
; CHECK: store i64 %this, ptr %this.spill.addr
; CHECK: ret ptr %hdl

; See that %this was loaded from the frame
; CHECK-LABEL: @f_direct.resume(
; CHECK:  %this.reload = load i64, ptr %this.reload.addr
; CHECK:  call void @print2(i64 %this.reload)
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

declare noalias ptr @myAlloc(i64, i32)
declare double @print(double)
declare void @print2(i64)
declare void @free(ptr)
