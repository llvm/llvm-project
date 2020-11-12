; Tests that allocas after coro.begin are properly that do not need to
; live on the frame are properly moved to the .resume function.
; RUN: opt < %s -coro-split -S | FileCheck %s
; RUN: opt < %s -passes=coro-split -S | FileCheck %s

define i8* @f() "coroutine.presplit"="1" {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  %x = alloca i32
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
  i8 1, label %cleanup]
resume:
  %x.value = load i32, i32* %x
  call void @print(i32 %x.value)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl
}

; CHECK-LABEL: @f.resume(
; CHECK-NEXT:  entry.resume:
; CHECK-NEXT:    [[VFRAME:%.*]] = bitcast %f.Frame* [[FRAMEPTR:%.*]] to i8*
; CHECK-NEXT:    [[X:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[X_VALUE:%.*]] = load i32, i32* [[X]], align 4
; CHECK-NEXT:    call void @print(i32 [[X_VALUE]])
; CHECK-NEXT:    call void @free(i8* [[VFRAME]])
; CHECK-NEXT:    ret void

declare i8* @llvm.coro.free(token, i8*)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i1 @llvm.coro.alloc(token)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)

declare void @print(i32)
declare noalias i8* @malloc(i32)
declare void @free(i8*)
