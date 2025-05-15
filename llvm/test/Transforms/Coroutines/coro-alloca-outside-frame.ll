; Tests that CoroSplit can succesfully skip allocas that shall not live on the frame
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S -o %t.ll
; RUN: FileCheck --input-file=%t.ll %s

; %y and %alias_phi would all go to the frame, but not %x
; CHECK:       %g.Frame = type { ptr, ptr, i64, ptr, i1 }

; CHECK-LABEL: @g(
; CHECK:         %x = alloca i64, align 8
; CHECK-NOT:     %x.reload.addr = getelementptr inbounds %g.Frame, ptr %hdl, i32 0, i32 2
; CHECK:         %y.reload.addr = getelementptr inbounds %g.Frame, ptr %hdl, i32 0, i32 2
; CHECK:         %alias_phi = phi ptr [ %y.reload.addr, %merge.from.flag_false ], [ %x, %entry ]


; The deprecated !coro.outside.frame metadata is parsed but doesn't do anything.
define ptr @f() {
entry:
  %x = alloca i64, !coro.outside.frame !{}
  ret ptr %x
}


define ptr @g(i1 %n) presplitcoroutine {
entry:
  %x = alloca i64
  %y = alloca i64
  call void @llvm.coro.outside.frame(ptr %x)
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
  %alias_phi = phi ptr [ %x, %flag_true ], [ %y, %flag_false ]
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  call void @print(ptr %alias_phi)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}


declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1, token)

declare void @print(ptr)
declare noalias ptr @malloc(i32)
declare void @free(ptr)
