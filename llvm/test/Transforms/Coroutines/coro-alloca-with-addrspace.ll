; RUN: opt < %s -passes='cgscc(coro-split)' -S | FileCheck %s

define ptr @f(i1 %n) presplitcoroutine {
entry:
  %x = alloca i64, addrspace(5)
  %y = alloca i64, addrspace(5)
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  br i1 %n, label %flag_true, label %flag_false

flag_true:
  br label %merge

flag_false:
  br label %merge

merge:
  %alias_phi = phi ptr addrspace(5) [ %x, %flag_true ], [ %y, %flag_false ]
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  call void @print(ptr addrspace(5) %alias_phi)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0)
  ret ptr %hdl
}

; CHECK-LABEL: @f(
; CHECK:  [[X_ADDR:%[0-9]+]] = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 2
; CHECK:  %x.reload.addr = addrspacecast ptr [[X_ADDR]] to ptr addrspace(5)
; CHECK:  [[Y_ADDR:%[0-9]+]] = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 3
; CHECK:  %y.reload.addr = addrspacecast ptr [[Y_ADDR]] to ptr addrspace(5)

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1)

declare void @print(ptr)
declare noalias ptr @malloc(i32)
declare void @free(ptr)
