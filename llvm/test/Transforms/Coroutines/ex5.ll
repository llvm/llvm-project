; Fifth example from Doc/Coroutines.rst (final suspend)
; RUN: opt < %s -aa-pipeline=basic-aa -passes='default<O2>' -preserve-alignment-assumptions-during-inlining=false -S | FileCheck %s

define ptr @f(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call noalias ptr @llvm.coro.begin(token %id, ptr %alloc)
  br label %while.cond
while.cond:
  %n.val = phi i32 [ %n, %entry ], [ %dec, %while.body ]
  %cmp = icmp sgt i32 %n.val, 0
  br i1 %cmp, label %while.body, label %while.end

while.body:
  %dec = add nsw i32 %n.val, -1
  call void @print(i32 %n.val) #4
  %s = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %s, label %suspend [i8 0, label %while.cond
                                i8 1, label %cleanup]
while.end:
  %s.final = call i8 @llvm.coro.suspend(token none, i1 true)
  switch i8 %s.final, label %suspend [i8 0, label %trap
                                      i8 1, label %cleanup]
trap: 
  call void @llvm.trap()
  unreachable
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret ptr %hdl
}

declare noalias ptr @malloc(i32)
declare void @print(i32)
declare void @llvm.trap()
declare void @free(ptr nocapture)

declare token @llvm.coro.id( i32, ptr, ptr, ptr)
declare i32 @llvm.coro.size.i32()
declare ptr @llvm.coro.begin(token, ptr)
declare token @llvm.coro.save(ptr)
declare i8 @llvm.coro.suspend(token, i1)
declare ptr @llvm.coro.free(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

; CHECK-LABEL: @main
define i32 @main() {
entry:
  %hdl = call ptr @f(i32 4)
  br label %while
while:
  call void @llvm.coro.resume(ptr %hdl)
  %done = call i1 @llvm.coro.done(ptr %hdl)
  br i1 %done, label %end, label %while
end:
  call void @llvm.coro.destroy(ptr %hdl)
  ret i32 0

; CHECK:      call void @print(i32 4)
; CHECK:      call void @print(i32 3)
; CHECK:      call void @print(i32 2)
; CHECK:      call void @print(i32 1)
; CHECK:      ret i32 0
}

declare i1 @llvm.coro.done(ptr)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)
