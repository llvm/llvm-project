; Check that coro-split doesn't choke on intrinsics in unreachable blocks
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S

define ptr @f(i1 %arg) presplitcoroutine personality i32 0 {
entry:
  %arg.addr = alloca i1
  store i1 %arg, ptr %arg.addr
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  br label %cont

cont:
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl

no.predecessors:
  %argval = load i1, ptr %arg.addr
  call void @print(i1 %argval)
  br label %suspend

}

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare noalias ptr @malloc(i32)
declare void @print(i1)
declare void @free(ptr)
