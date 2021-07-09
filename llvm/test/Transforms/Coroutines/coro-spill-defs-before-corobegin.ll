; Verifies that phi and invoke definitions before CoroBegin are spilled properly.
; RUN: opt < %s -passes='cgscc(coro-split),simplify-cfg,early-cse,simplify-cfg' -S | FileCheck %s

define i8* @f(i1 %n) "coroutine.presplit"="1" personality i32 0 {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %flag = call i1 @check(i8* %alloc)
  br i1 %flag, label %flag_true, label %flag_false

flag_true:
  br label %merge

flag_false:
  br label %merge

merge:
  %value_phi = phi i32 [ 0, %flag_true ], [ 1, %flag_false ]
  %value_invoke = invoke i32 @calc() to label %normal unwind label %lpad

normal:
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  call i32 @print(i32 %value_phi)
  call i32 @print(i32 %value_invoke)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  call i32 @print(i32 %value_phi)
  call i32 @print(i32 %value_invoke)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl

lpad:
  %lpval = landingpad { i8*, i32 }
     cleanup

  resume { i8*, i32 } %lpval
}

; Verifies that the both value_phi and value_invoke are stored correctly in the coroutine frame
; CHECK: %f.Frame = type { void (%f.Frame*)*, void (%f.Frame*)*, i32, i32, i1 }
; CHECK-LABEL: @f(
; CHECK:       %alloc = call i8* @malloc(i32 32)
; CHECK-NEXT:  %flag = call i1 @check(i8* %alloc)
; CHECK-NEXT:  %spec.select = select i1 %flag, i32 0, i32 1
; CHECK-NEXT:  %value_invoke = call i32 @calc()
; CHECK-NEXT:  %hdl = call noalias nonnull i8* @llvm.coro.begin(token %id, i8* %alloc)

; CHECK:       store void (%f.Frame*)* @f.destroy, void (%f.Frame*)** %destroy.addr
; CHECK-NEXT:  %value_invoke.spill.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 3
; CHECK-NEXT:  store i32 %value_invoke, i32* %value_invoke.spill.addr
; CHECK-NEXT:  %value_phi.spill.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 2
; CHECK-NEXT:  store i32 %spec.select, i32* %value_phi.spill.addr

declare i8* @llvm.coro.free(token, i8*)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i1 @llvm.coro.alloc(token)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)

declare noalias i8* @malloc(i32)
declare i32 @print(i32)
declare i1 @check(i8*)
declare i32 @calc()
declare void @free(i8*)
