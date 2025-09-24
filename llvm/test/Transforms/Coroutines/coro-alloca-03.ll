; Tests that allocas escaped through function calls will live on the frame.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @f() presplitcoroutine {
entry:
  %x = alloca i64
  %y = alloca i64
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  call void @capture_call(ptr %x)
  call void @nocapture_call(ptr %y)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; %x needs to go to the frame since it's escaped; %y will stay as local since it doesn't escape.
; CHECK:        %f.Frame = type { ptr, ptr, i64, i1 }
; CHECK-LABEL:  define ptr @f()
; CHECK:          %y = alloca i64, align 8
; CHECK:          %x.reload.addr = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 2
; CHECK:          call void @capture_call(ptr %x.reload.addr)
; CHECK:          call void @nocapture_call(ptr %y)

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare void @capture_call(ptr)
declare void @nocapture_call(ptr nocapture)
declare noalias ptr @malloc(i32)
declare void @free(ptr)
