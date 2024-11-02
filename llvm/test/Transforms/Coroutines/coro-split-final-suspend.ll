; Tests that we'll generate the store to the final suspend index if we see the unwind coro end.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @unwind_coro_end() presplitcoroutine personality i32 3 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  call void @print(i32 0)
  br label %init

init:
  %initial_suspend = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %initial_suspend, label %init_suspend [i8 0, label %init_resume
                                                   i8 1, label %init_suspend]

init_suspend:
  ret ptr %hdl

init_resume:
  br label %susp

susp:
  %0 = call i8 @llvm.coro.suspend(token none, i1 true)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %suspend]
resume:
  invoke void @print(i32 1) to label %suspend unwind label %lpad

suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 false)
  call void @print(i32 0)
  ret ptr %hdl

lpad:
  %lpval = landingpad { ptr, i32 }
     cleanup

  call void @print(i32 2)
  %need.resume = call i1 @llvm.coro.end(ptr null, i1 true)
  br i1 %need.resume, label %eh.resume, label %cleanup.cont

cleanup.cont:
  call void @print(i32 3)
  br label %eh.resume

eh.resume:
  resume { ptr, i32 } %lpval
}

; Tests that we need to store the final index if we see unwind coro end.
; CHECK: define{{.*}}@unwind_coro_end.resume
; CHECK: store i1 true, ptr %index.addr

; Tests the use of final index in the destroy function.
; CHECK: define{{.*}}@unwind_coro_end.destroy
; CHECK: %[[INDEX:.+]] = load i1, ptr %index.addr
; CHECK-NEXT: switch i1 %[[INDEX]],

define ptr @nounwind_coro_end(i1 %val) presplitcoroutine personality i32 3 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  call void @print(i32 0)
  br label %init

init:
  %initial_suspend = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %initial_suspend, label %init_suspend [i8 0, label %init_resume
                                                   i8 1, label %init_suspend]

init_suspend:
  ret ptr %hdl

init_resume:
  br label %susp

susp:
  %0 = call i8 @llvm.coro.suspend(token none, i1 true)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %suspend]
resume:
  call void @print(i32 1)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 false)
  call void @print(i32 0)
  ret ptr %hdl
}

; Tests that we can omit to store the final suspend index if we don't
; see unwind coro end.
; CHECK: define{{.*}}@nounwind_coro_end.resume
; CHECK-NOT: store i1 true, ptr %index.addr
; CHECK: }

; Tests that we judge the final suspend case by the nullness of resume function.
; CHECK: define{{.*}}@nounwind_coro_end.destroy
; CHECK:  %[[RESUME_FN:.+]] = load ptr, ptr %hdl, align 8
; CHECK:  %{{.*}} = icmp eq ptr %[[RESUME_FN]], null

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1)

declare noalias ptr @malloc(i32)
declare void @print(i32)
declare void @free(ptr)
