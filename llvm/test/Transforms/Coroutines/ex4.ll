; Fourth example from Doc/Coroutines.rst (coroutine promise)
; RUN: opt < %s -passes='default<O2>' -S | FileCheck %s

define i8* @f(i32 %n) presplitcoroutine {
entry:
  %promise = alloca i32
  %pv = bitcast i32* %promise to i8*
  %id = call token @llvm.coro.id(i32 0, i8* %pv, i8* null, i8* null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %phi)
  br label %loop
loop:
  %n.val = phi i32 [ %n, %coro.begin ], [ %inc, %loop ]
  %inc = add nsw i32 %n.val, 1
  store i32 %n.val, i32* %promise
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %loop
                                i8 1, label %cleanup]
cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 false)
  ret i8* %hdl
}

; CHECK-LABEL: @main
define i32 @main() {
entry:
  %hdl = call i8* @f(i32 4)
  %promise.addr.raw = call i8* @llvm.coro.promise(i8* %hdl, i32 4, i1 false)
  %promise.addr = bitcast i8* %promise.addr.raw to i32*
  %val0 = load i32, i32* %promise.addr
  call void @print(i32 %val0)
  call void @llvm.coro.resume(i8* %hdl)
  %val1 = load i32, i32* %promise.addr
  call void @print(i32 %val1)
  call void @llvm.coro.resume(i8* %hdl)
  %val2 = load i32, i32* %promise.addr
  call void @print(i32 %val2)
  call void @llvm.coro.destroy(i8* %hdl)
  ret i32 0
; CHECK:      call void @print(i32 4)
; CHECK-NEXT: call void @print(i32 5)
; CHECK-NEXT: call void @print(i32 6)
; CHECK:      ret i32 0
}

declare i8* @llvm.coro.promise(i8*, i32, i1)
declare i8* @malloc(i32)
declare void @free(i8*)
declare void @print(i32)

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i1 @llvm.coro.alloc(token)
declare i32 @llvm.coro.size.i32()
declare i8* @llvm.coro.begin(token, i8*)
declare i8 @llvm.coro.suspend(token, i1)
declare i8* @llvm.coro.free(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)

declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)
