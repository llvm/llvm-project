; Third example from Doc/Coroutines.rst (two suspend points)
; RUN: opt < %s -aa-pipeline=basic-aa -passes='default<O2>' -S | FileCheck %s

define ptr @f(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %alloc = call ptr @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias ptr @llvm.coro.begin(token %id, ptr %phi)
  br label %loop
loop:
  %n.val = phi i32 [ %n, %coro.begin ], [ %inc, %loop.resume ]
  call void @print(i32 %n.val) #4
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %loop.resume
                                i8 1, label %cleanup]
loop.resume:
  %inc = add nsw i32 %n.val, 1
  %sub = xor i32 %n.val, -1
  call void @print(i32 %sub)
  %1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %1, label %suspend [i8 0, label %loop
                                i8 1, label %cleanup]
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 false)
  ret ptr %hdl
}

; CHECK-LABEL: @main
define i32 @main() {
entry:
  %hdl = call ptr @f(i32 4)
  call void @llvm.coro.resume(ptr %hdl)
  call void @llvm.coro.resume(ptr %hdl)
  %c = ptrtoint ptr %hdl to i64
  %to = icmp eq i64 %c, 0
  br i1 %to, label %return, label %destroy
destroy:
  call void @llvm.coro.destroy(ptr %hdl)
  br label %return
return:
  ret i32 0
; CHECK-NOT:  ptr @malloc
; CHECK:      call void @print(i32 4)
; CHECK-NEXT: call void @print(i32 -5)
; CHECK-NEXT: call void @print(i32 5)
; CHECK:      ret i32 0
}

declare ptr @malloc(i32)
declare void @free(ptr)
declare void @print(i32)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare i32 @llvm.coro.size.i32()
declare ptr @llvm.coro.begin(token, ptr)
declare i8 @llvm.coro.suspend(token, i1)
declare ptr @llvm.coro.free(token, ptr)
declare i1 @llvm.coro.end(ptr, i1)

declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)
