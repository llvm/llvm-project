; Check that we create copy the data from the alloca into the coroutine
; frame slot if it was written to.
; RUN: opt < %s -coro-split -S | FileCheck %s
; RUN: opt < %s -passes=coro-split -S | FileCheck %s

define i8* @f() "coroutine.presplit"="1" {
entry:
  %x.addr = alloca i64
  call void @use(i64* %x.addr) ; might write to %x
  %y.addr = alloca i64
  %y = load i64, i64* %y.addr ; cannot modify the value, don't need to copy
  call void @print(i64 %y)

  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @myAlloc(i64 %y, i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call void @use(i64* %x.addr)
  call void @use(i64* %y.addr)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl
}

; See that we added both x and y to the frame.
; CHECK: %f.Frame = type { void (%f.Frame*)*, void (%f.Frame*)*, i64, i64, i1 }

; See that all of the uses prior to coro-begin stays put.
; CHECK-LABEL: define i8* @f() {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %x.addr = alloca i64
; CHECK-NEXT:   call void @use(i64* %x.addr)
; CHECK-NEXT:   %y.addr = alloca i64
; CHECK-NEXT:   %y = load i64, i64* %y.addr
; CHECK-NEXT:   call void @print(i64 %y)

; See that we only copy the x as y was not modified prior to coro.begin.
; CHECK:  store void (%f.Frame*)* @f.destroy, void (%f.Frame*)** %destroy.addr
; CHECK-NEXT:  %0 = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 2
; CHECK-NEXT:  %1 = load i64, i64* %x.addr
; CHECK-NEXT:  store i64 %1, i64* %0
; CHECK-NEXT:  %index.addr1 = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 4
; CHECK-NEXT:  store i1 false, i1* %index.addr1
; CHECK-NEXT:  ret i8* %hdl

declare i8* @llvm.coro.free(token, i8*)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i1 @llvm.coro.alloc(token)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)

declare noalias i8* @myAlloc(i64, i32)
declare void @print(i64)
declare void @use(i64*)
declare void @free(i8*)
