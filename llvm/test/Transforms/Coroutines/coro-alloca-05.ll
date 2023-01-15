; Tests that allocas after coro.begin are properly that do not need to
; live on the frame are properly moved to the .resume function.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @f() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %x = alloca i32
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
  i8 1, label %cleanup]
resume:
  %x.value = load i32, ptr %x
  call void @print(i32 %x.value)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0)
  ret ptr %hdl
}

; CHECK-LABEL: @f.resume(
; CHECK-NEXT:  entry.resume:
; CHECK-NEXT:    [[X:%.*]] = alloca i32, align 4
; CHECK:         [[X_VALUE:%.*]] = load i32, ptr [[X]], align 4
; CHECK-NEXT:    call void @print(i32 [[X_VALUE]])
; CHECK:         call void @free(ptr [[FRAMEPTR:%.*]])
; CHECK:         ret void

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1)

declare void @print(i32)
declare noalias ptr @malloc(i32)
declare void @free(ptr)
