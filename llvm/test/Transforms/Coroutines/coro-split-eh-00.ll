; Tests that coro-split removes cleanup code after coro.end in resume functions
; and retains it in the start function.
; RUN: opt < %s -coro-split -S | FileCheck %s
; RUN: opt < %s -passes=coro-split -S | FileCheck %s

define i8* @f(i1 %val) "coroutine.presplit"="1" personality i32 3 {
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
  %lpval = landingpad { i8*, i32 }
     cleanup

  call void @print(i32 2)
  %need.resume = call i1 @llvm.coro.end(i8* null, i1 true)
  br i1 %need.resume, label %eh.resume, label %cleanup.cont

cleanup.cont:
  call void @print(i32 3) ; should not be present in f.resume
  br label %eh.resume

eh.resume:
  resume { i8*, i32 } %lpval
}

; Verify that start function contains both print calls the one before and after coro.end
; CHECK-LABEL: define i8* @f(
; CHECK: invoke void @print(i32 1)
; CHECK:   to label %AfterCoroEnd unwind label %lpad

; CHECK: AfterCoroEnd:
; CHECK:   call void @print(i32 0)
; CHECK:   ret i8* %hdl

; CHECK:         lpad:
; CHECK-NEXT:      %lpval = landingpad { i8*, i32 }
; CHECK-NEXT:         cleanup
; CHECK-NEXT:      call void @print(i32 2)
; CHECK-NEXT:      call void @print(i32 3)
; CHECK-NEXT:      resume { i8*, i32 } %lpval

; VERIFY Resume Parts

; Verify that resume function does not contains both print calls appearing after coro.end
; CHECK-LABEL: define internal fastcc void @f.resume
; CHECK: invoke void @print(i32 1)
; CHECK:   to label %CoroEnd unwind label %lpad

; CHECK:      CoroEnd:
; CHECK-NEXT:   ret void

; CHECK:         lpad:
; CHECK-NEXT:      %lpval = landingpad { i8*, i32 }
; CHECK-NEXT:         cleanup
; CHECK-NEXT:      call void @print(i32 2)
; CHECK-NEXT:      resume { i8*, i32 } %lpval

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

