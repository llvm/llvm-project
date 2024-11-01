; Verifies that phi and invoke definitions before CoroBegin are spilled properly.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse,simplifycfg' -S | FileCheck %s

define ptr @f(i1 %n) presplitcoroutine personality i32 0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %flag = call i1 @check(ptr %alloc)
  br i1 %flag, label %flag_true, label %flag_false

flag_true:
  br label %merge

flag_false:
  br label %merge

merge:
  %value_phi = phi i32 [ 0, %flag_true ], [ 1, %flag_false ]
  %value_invoke = invoke i32 @calc() to label %normal unwind label %lpad

normal:
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
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
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0)
  ret ptr %hdl

lpad:
  %lpval = landingpad { ptr, i32 }
     cleanup

  resume { ptr, i32 } %lpval
}

; Verifies that the both value_phi and value_invoke are stored correctly in the coroutine frame
; CHECK: %f.Frame = type { ptr, ptr, i32, i32, i1 }
; CHECK-LABEL: @f(
; CHECK:       %alloc = call ptr @malloc(i32 32)
; CHECK-NEXT:  %flag = call i1 @check(ptr %alloc)
; CHECK-NEXT:  %spec.select = select i1 %flag, i32 0, i32 1
; CHECK-NEXT:  %value_invoke = call i32 @calc()
; CHECK-NEXT:  %hdl = call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %alloc)

; CHECK:       store ptr @f.destroy, ptr %destroy.addr
; CHECK-NEXT:  %value_invoke.spill.addr = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 3
; CHECK-NEXT:  store i32 %value_invoke, ptr %value_invoke.spill.addr
; CHECK-NEXT:  %value_phi.spill.addr = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 2
; CHECK-NEXT:  store i32 %spec.select, ptr %value_phi.spill.addr

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1)

declare noalias ptr @malloc(i32)
declare i32 @print(i32)
declare i1 @check(ptr)
declare i32 @calc()
declare void @free(ptr)
