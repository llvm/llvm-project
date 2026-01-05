; First example from Doc/Coroutines.rst (one block loop)
; RUN: opt < %s -aa-pipeline=basic-aa -passes='default<O2>' -preserve-alignment-assumptions-during-inlining=false -S | FileCheck %s

define ptr @f(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call noalias ptr @llvm.coro.begin(token %id, ptr %alloc)
  br label %loop
loop:
  %n.val = phi i32 [ %n, %entry ], [ %inc, %loop ]
  %inc = add nsw i32 %n.val, 1
  call void @print(i32 %n.val)
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

; CHECK-LABEL: @main(
define i32 @main() {
entry:
  %hdl = call ptr @f(i32 4)
  call void @llvm.coro.resume(ptr %hdl)
  call void @llvm.coro.resume(ptr %hdl)
  call void @llvm.coro.destroy(ptr %hdl)
  ret i32 0
; CHECK-NEXT: entry:
; CHECK:      call void @print(i32 4)
; CHECK:      call void @print(i32 5)
; CHECK:      call void @print(i32 6)
; CHECK:      ret i32 0
}

declare ptr @malloc(i32)
declare void @free(ptr)
declare void @print(i32)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i32 @llvm.coro.size.i32()
declare ptr @llvm.coro.begin(token, ptr)
declare i8 @llvm.coro.suspend(token, i1)
declare ptr @llvm.coro.free(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)
