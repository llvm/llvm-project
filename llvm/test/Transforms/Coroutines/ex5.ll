; Fifth example from Doc/Coroutines.rst (final suspend)
; RUN: opt < %s -aa-pipeline=basic-aa -passes='default<O2>' -preserve-alignment-assumptions-during-inlining=false -S | FileCheck %s

define i8* @f(i32 %n) "coroutine.presplit"="0" {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %alloc)
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
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 false)
  ret i8* %hdl
}

declare noalias i8* @malloc(i32)
declare void @print(i32)
declare void @llvm.trap()
declare void @free(i8* nocapture)

declare token @llvm.coro.id( i32, i8*, i8*, i8*)
declare i32 @llvm.coro.size.i32()
declare i8* @llvm.coro.begin(token, i8*)
declare token @llvm.coro.save(i8*)
declare i8 @llvm.coro.suspend(token, i1)
declare i8* @llvm.coro.free(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)

; CHECK-LABEL: @main
define i32 @main() {
entry:
  %hdl = call i8* @f(i32 4)
  br label %while
while:
  call void @llvm.coro.resume(i8* %hdl)
  %done = call i1 @llvm.coro.done(i8* %hdl)
  br i1 %done, label %end, label %while
end:
  call void @llvm.coro.destroy(i8* %hdl)
  ret i32 0

; CHECK:      call void @print(i32 4)
; CHECK:      call void @print(i32 3)
; CHECK:      call void @print(i32 2)
; CHECK:      call void @print(i32 1)
; CHECK:      ret i32 0
}

declare i1 @llvm.coro.done(i8*)
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)
