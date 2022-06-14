; Second example from Doc/Coroutines.rst (custom alloc and free functions)
; RUN: opt < %s -passes='default<O2>' -S | FileCheck %s

define i8* @f(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @CustomAlloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %phi)
  br label %loop
loop:
  %n.val = phi i32 [ %n, %coro.begin ], [ %inc, %loop ]
  %inc = add nsw i32 %n.val, 1
  call void @print(i32 %n.val)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %loop
                                i8 1, label %cleanup]
cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  %need.dyn.free = icmp ne i8* %mem, null
  br i1 %need.dyn.free, label %dyn.free, label %suspend
dyn.free:
  call void @CustomFree(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 false)
  ret i8* %hdl
}

; CHECK-LABEL: @main
define i32 @main() {
entry:
  %hdl = call i8* @f(i32 4)
  call void @llvm.coro.resume(i8* %hdl)
  call void @llvm.coro.resume(i8* %hdl)
  %to = icmp eq i8* %hdl, null
  br i1 %to, label %return, label %destroy
destroy:
  call void @llvm.coro.destroy(i8* %hdl)
  br label %return
return:
  ret i32 0
; CHECK-NOT:  call i8* @CustomAlloc
; CHECK:      call void @print(i32 4)
; CHECK-NEXT: call void @print(i32 5)
; CHECK-NEXT: call void @print(i32 6)
; CHECK-NEXT: ret i32 0
}

declare i8* @CustomAlloc(i32)
declare void @CustomFree(i8*)
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
