; Tests that coro-split pass splits the coroutine into f, f.resume and f.destroy
; RUN: opt < %s -coro-split -S | FileCheck %s

define i8* @f() "coroutine.presplit"="1" {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  call void @print(i32 0)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume 
                                i8 1, label %cleanup]
resume:
  call void @print(i32 1)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(i8* %hdl, i1 0)  
  ret i8* %hdl
}

; CHECK-LABEL: @f(
; CHECK: call i8* @malloc
; CHECK: call void @print(i32 0)
; CHECK-NOT: call void @print(i32 1)
; CHECK-NOT: call void @free(
; CHECK: ret i8* %hdl

; CHECK-LABEL: @f.resume(
; CHECK-NOT: call i8* @malloc
; CHECK-NOT: call void @print(i32 0)
; CHECK: call void @print(i32 1)
; CHECK-NOT: call void @print(i32 0)
; CHECK: call void @free(
; CHECK: ret void

; CHECK-LABEL: @f.destroy(
; CHECK-NOT: call i8* @malloc
; CHECK-NOT: call void @print(
; CHECK: call void @free(
; CHECK: ret void

declare i8* @llvm.coro.free(token, i8*)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i8* @llvm.coro.alloc(token)
declare i8* @llvm.coro.begin(token, i8*)
declare void @llvm.coro.end(i8*, i1) 

declare noalias i8* @malloc(i32)
declare void @print(i32)
declare void @free(i8*)
