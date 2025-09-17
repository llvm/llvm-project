; Tests that coro-split removes cleanup code after coro.end in resume functions
; and retains it in the start function.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @f(i1 %val) presplitcoroutine personality i32 3 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  call void @print(i32 0)
  br i1 %val, label %resume, label %susp

susp:
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %suspend]
resume:
  invoke void @print(i32 1) to label %suspend unwind label %lpad

suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  call void @print(i32 0) ; should not be present in f.resume
  ret ptr %hdl

lpad:
  %lpval = landingpad { ptr, i32 }
     cleanup

  call void @print(i32 2)
  %need.resume = call i1 @llvm.coro.end(ptr null, i1 true, token none)
  br i1 %need.resume, label %eh.resume, label %cleanup.cont

cleanup.cont:
  call void @print(i32 3) ; should not be present in f.resume
  br label %eh.resume

eh.resume:
  resume { ptr, i32 } %lpval
}

; Verify that start function contains both print calls the one before and after coro.end
; CHECK-LABEL: define ptr @f(
; CHECK: invoke void @print(i32 1)
; CHECK:   to label %AfterCoroEnd unwind label %lpad

; CHECK: AfterCoroEnd:
; CHECK:   call void @print(i32 0)
; CHECK:   ret ptr %hdl

; CHECK:         lpad:
; CHECK-NEXT:      %lpval = landingpad { ptr, i32 }
; CHECK-NEXT:         cleanup
; CHECK-NEXT:      call void @print(i32 2)
; CHECK-NEXT:      call void @print(i32 3)
; CHECK-NEXT:      resume { ptr, i32 } %lpval

; VERIFY Resume Parts

; Verify that resume function does not contains both print calls appearing after coro.end
; CHECK-LABEL: define internal fastcc void @f.resume
; CHECK: invoke void @print(i32 1)
; CHECK:   to label %CoroEnd unwind label %lpad

; CHECK:      CoroEnd:
; CHECK-NEXT:   ret void

; CHECK:         lpad:
; CHECK-NEXT:      %lpval = landingpad { ptr, i32 }
; CHECK-NEXT:         cleanup
; CHECK-NEXT:      call void @print(i32 2)
; Checks that the coroutine would be marked as done if it exits in unwinding path.
; CHECK-NEXT:      store ptr null, ptr %hdl, align 8
; CHECK-NEXT:      resume { ptr, i32 } %lpval

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1, token)

declare noalias ptr @malloc(i32)
declare void @print(i32)
declare void @free(ptr)
