; RUN: opt < %s -passes='module(coro-early,coro-early)' -S | FileCheck %s

; Check that coro-early can run twice without asserting/crashing.

; CHECK-LABEL: define ptr @f
; CHECK: call token @llvm.coro.id(i32 0, ptr null, ptr @f, ptr null)

define ptr @f(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)

  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i32 @llvm.coro.size.i32()
declare noalias ptr @malloc(i32)
declare ptr @llvm.coro.begin(token, ptr)
declare i8  @llvm.coro.suspend(token, i1)
declare ptr @llvm.coro.free(token, ptr)
declare void @free(ptr)
declare void @llvm.coro.end(ptr, i1, token)
