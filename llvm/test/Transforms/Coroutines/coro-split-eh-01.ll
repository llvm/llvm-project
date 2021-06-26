; Tests that coro-split removes cleanup code after coro.end in resume functions
; and retains it in the start function.
; RUN: opt < %s -passes=coro-split -S | FileCheck %s

define i8* @f2(i1 %val) "coroutine.presplit"="1" personality i32 4 {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  call void @print(i32 0)
  br i1 %val, label %resume, label %susp

susp:
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %suspend]
resume:
  invoke void @print(i32 1) to label %suspend unwind label %lpad

suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  call void @print(i32 0) ; should not be present in f.resume
  ret i8* %hdl

lpad:
  %tok = cleanuppad within none []
  call void @print(i32 2)
  %unused = call i1 @llvm.coro.end(i8* null, i1 true) [ "funclet"(token %tok) ]
  cleanupret from %tok unwind label %cleanup.cont

cleanup.cont:
  %tok2 = cleanuppad within none []
  call void @print(i32 3) ; should not be present in f.resume
  cleanupret from %tok2 unwind to caller
}

; Verify that start function contains both print calls the one before and after coro.end
; CHECK-LABEL: define i8* @f2(
; CHECK: invoke void @print(i32 1)
; CHECK:   to label %AfterCoroEnd unwind label %lpad

; CHECK: AfterCoroEnd:
; CHECK:   call void @print(i32 0)
; CHECK:   ret i8* %hdl

; CHECK:      lpad:
; CHECK-NEXT:   %tok = cleanuppad within none []
; CHECK-NEXT:   call void @print(i32 2)
; CHECK-NEXT:   call void @print(i32 3)
; CHECK-NEXT:   cleanupret from %tok unwind to caller

; VERIFY Resume Parts

; Verify that resume function does not contains both print calls appearing after coro.end
; CHECK-LABEL: define internal fastcc void @f2.resume
; CHECK: invoke void @print(i32 1)
; CHECK:   to label %CoroEnd unwind label %lpad

; CHECK:      CoroEnd:
; CHECK-NEXT:   ret void

; CHECK:      lpad:
; CHECK-NEXT:   %tok = cleanuppad within none []
; CHECK-NEXT:   call void @print(i32 2)
; CHECK-NEXT:   cleanupret from %tok unwind to caller

declare i8* @llvm.coro.free(token, i8*)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i8* @llvm.coro.alloc(token)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)

declare noalias i8* @malloc(i32)
declare void @print(i32)
declare void @free(i8*)

