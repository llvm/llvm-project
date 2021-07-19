; Check that we create copy the data from the alloca into the coroutine
; frame slot if it was written to.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define i8* @f() "coroutine.presplit"="1" {
entry:
  %a.addr = alloca i64 ; read-only before coro.begin
  %a = load i64, i64* %a.addr ; cannot modify the value, don't need to copy

  %x.addr = alloca i64
  call void @use(i64* %x.addr) ; uses %x.addr before coro.begin

  %y.addr = alloca i64
  %y.cast = bitcast i64* %y.addr to i8* ; alias created and used after coro.begin
  
  %z.addr = alloca i64
  %flag = call i1 @check()
  br i1 %flag, label %flag_true, label %flag_merge

flag_true:
  call void @use(i64* %z.addr) ; conditionally used %z.addr
  br label %flag_merge

flag_merge:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @myAlloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  call void @llvm.memset.p0i8.i32(i8* %y.cast, i8 1, i32 4, i1 false)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call void @use(i64* %a.addr)
  call void @use(i64* %x.addr)
  call void @use(i64* %y.addr)
  call void @use(i64* %z.addr)
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
; CHECK: %f.Frame = type { void (%f.Frame*)*, void (%f.Frame*)*, i64, i64, i64, i64, i1 }

; See that all of the uses prior to coro-begin stays put.
; CHECK-LABEL: define i8* @f() {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a.addr = alloca i64
; CHECK-NEXT:   %x.addr = alloca i64
; CHECK-NEXT:   call void @use(i64* %x.addr)
; CHECK-NEXT:   %y.addr = alloca i64
; CHECK-NEXT:   %z.addr = alloca i64

; See that we only copy the x as y was not modified prior to coro.begin.
; CHECK:       store void (%f.Frame*)* @f.destroy, void (%f.Frame*)** %destroy.addr
; The next 3 instructions are to copy data in %x.addr from stack to frame.
; CHECK-NEXT:  %0 = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 3
; CHECK-NEXT:  %1 = load i64, i64* %x.addr, align 4
; CHECK-NEXT:  store i64 %1, i64* %0, align 4
; The next 2 instructions are to recreate %y.cast in the original IR.
; CHECK-NEXT:  %2 = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 4
; CHECK-NEXT:  %3 = bitcast i64* %2 to i8*
; The next 3 instructions are to copy data in %z.addr from stack to frame.
; CHECK-NEXT:  %4 = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 5
; CHECK-NEXT:  %5 = load i64, i64* %z.addr, align 4
; CHECK-NEXT:  store i64 %5, i64* %4, align 4
; CHECK-NEXT:  call void @llvm.memset.p0i8.i32(i8* %3, i8 1, i32 4, i1 false)
; CHECK-NEXT:  %index.addr1 = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 6
; CHECK-NEXT:  store i1 false, i1* %index.addr1, align 1
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

declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i1)

declare noalias i8* @myAlloc(i32)
declare void @use(i64*)
declare void @free(i8*)
declare i1 @check()
