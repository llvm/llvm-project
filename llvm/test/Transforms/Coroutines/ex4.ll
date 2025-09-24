; Fourth example from Doc/Coroutines.rst (coroutine promise)
; RUN: opt < %s -passes='default<O2>' -S | FileCheck %s

define ptr @f(i32 %n) presplitcoroutine {
entry:
  %promise = alloca i32
  %id = call token @llvm.coro.id(i32 0, ptr %promise, ptr null, ptr null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias ptr @llvm.coro.begin(token %id, ptr %phi)
  br label %loop
loop:
  %n.val = phi i32 [ %n, %coro.begin ], [ %inc, %loop ]
  %inc = add nsw i32 %n.val, 1
  store i32 %n.val, ptr %promise
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %loop
                                i8 1, label %cleanup]
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret ptr %hdl
}

; CHECK-LABEL: @main
define i32 @main() {
entry:
  %hdl = call ptr @f(i32 4)
  %promise.addr.raw = call ptr @llvm.coro.promise(ptr %hdl, i32 4, i1 false)
  %val0 = load i32, ptr %promise.addr.raw
  call void @print(i32 %val0)
  call void @llvm.coro.resume(ptr %hdl)
  %val1 = load i32, ptr %promise.addr.raw
  call void @print(i32 %val1)
  call void @llvm.coro.resume(ptr %hdl)
  %val2 = load i32, ptr %promise.addr.raw
  call void @print(i32 %val2)
  call void @llvm.coro.destroy(ptr %hdl)
  ret i32 0
; CHECK:      call void @print(i32 4)
; CHECK-NEXT: call void @print(i32 5)
; CHECK-NEXT: call void @print(i32 6)
; CHECK:      ret i32 0
}

declare ptr @llvm.coro.promise(ptr, i32, i1)
declare ptr @malloc(i32)
declare void @free(ptr)
declare void @print(i32)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare i32 @llvm.coro.size.i32()
declare ptr @llvm.coro.begin(token, ptr)
declare i8 @llvm.coro.suspend(token, i1)
declare ptr @llvm.coro.free(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)
