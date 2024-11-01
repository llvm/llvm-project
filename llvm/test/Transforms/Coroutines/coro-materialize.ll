; Verifies that we materialize instruction across suspend points
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @f(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)

  %inc1 = add i32 %n, 1
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %inc2 = add i32 %inc1, 1
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]

resume2:
  call void @print(i32 %inc1)
  call void @print(i32 %inc2)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0)
  ret ptr %hdl
}

; See that we only spilled one value
; CHECK: %f.Frame = type { ptr, ptr, i32, i1 }
; CHECK-LABEL: @f(

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
declare void @print(i32)
declare void @free(ptr)
