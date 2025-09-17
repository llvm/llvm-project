; Verify that we correctly handle suspend when the coro.end block contains phi
; RUN: opt < %s -aa-pipeline=basic-aa -passes='default<O2>' -S | FileCheck %s

define ptr @f(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %cleanup i8 1, label %cleanup]

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  %r = phi i32 [%n, %entry], [1, %cleanup]
  call i1 @llvm.coro.end(ptr %hdl, i1 false, token none)  
  call void @print(i32 %r)
  ret ptr %hdl
}

; CHECK-LABEL: @main
define i32 @main() {
entry:
  %hdl = call ptr @f(i32 4)
  call void @llvm.coro.resume(ptr %hdl)
  ret i32 0
;CHECK: call void @print(i32 4)
;CHECK: ret i32 0
}

declare ptr @llvm.coro.alloc()
declare i32 @llvm.coro.size.i32()
declare ptr @llvm.coro.free(token, ptr)
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)
  
declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1, token) 

declare noalias ptr @malloc(i32)
declare void @print(i32)
declare void @free(ptr)
