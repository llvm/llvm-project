; Tests that a scalable alloca whose lifetime markers span a suspend point, but
; whose uses do not, stays on the stack instead of being put on the frame.
; RUN: opt < %s -passes='cgscc(coro-split)' -S | FileCheck %s

define ptr @f() presplitcoroutine {
entry:
  %pred = alloca <vscale x 16 x i1>
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  call void @llvm.lifetime.start.p0(ptr %pred)
  store <vscale x 16 x i1> zeroinitializer, ptr %pred
  %v = load <vscale x 16 x i1>, ptr %pred
  call void @use(<vscale x 16 x i1> %v)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  call void @llvm.lifetime.end.p0(ptr %pred)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; CHECK-LABEL: define ptr @f(
; CHECK:         %pred = alloca <vscale x 16 x i1>
; CHECK:         store <vscale x 16 x i1> zeroinitializer, ptr %pred
; CHECK:         %v = load <vscale x 16 x i1>, ptr %pred

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare void @llvm.lifetime.start.p0(ptr)
declare void @llvm.lifetime.end.p0(ptr)

declare void @use(<vscale x 16 x i1>)
declare noalias ptr @malloc(i32)
declare void @free(ptr)
