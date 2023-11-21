; Tests that if an alloca is escaped through storing the address,
; the alloac will be put on the frame.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @f() presplitcoroutine {
entry:
  %x = alloca i64
  %y = alloca ptr
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  store ptr %x, ptr %y
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  %x1 = load ptr, ptr %y
  call void @print(ptr %x1)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; CHECK:        %f.Frame = type { ptr, ptr, i64, ptr, i1 }
; CHECK-LABEL:  define ptr @f()
; CHECK:          %x.reload.addr = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 2
; CHECK:          %y.reload.addr = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 3
; CHECK:          store ptr %x.reload.addr, ptr %y.reload.addr, align 8

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1, token)

declare void @print(ptr)
declare noalias ptr @malloc(i32)
declare void @free(ptr)
