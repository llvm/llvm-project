; Check that we create copy the data from the alloca into the coroutine
; frame slot if it was written to.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @f() presplitcoroutine {
entry:
  %a.addr = alloca i64 ; read-only before coro.begin
  %a = load i64, ptr %a.addr ; cannot modify the value, don't need to copy

  %x.addr = alloca i64
  call void @use(ptr %x.addr) ; uses %x.addr before coro.begin

  %y.addr = alloca i64
  
  %z.addr = alloca i64
  %flag = call i1 @check()
  br i1 %flag, label %flag_true, label %flag_merge

flag_true:
  call void @use(ptr %z.addr) ; conditionally used %z.addr
  br label %flag_merge

flag_merge:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @myAlloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  call void @llvm.memset.p0.i32(ptr %y.addr, i8 1, i32 4, i1 false)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call void @use(ptr %a.addr)
  call void @use(ptr %x.addr)
  call void @use(ptr %y.addr)
  call void @use(ptr %z.addr)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; See that we added both x and y to the frame.
; CHECK: %f.Frame = type { ptr, ptr, i64, i64, i64, i64, i1 }

; See that all of the uses prior to coro-begin stays put.
; CHECK-LABEL: define ptr @f() {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a.addr = alloca i64
; CHECK-NEXT:   %x.addr = alloca i64
; CHECK-NEXT:   call void @use(ptr %x.addr)
; CHECK-NEXT:   %z.addr = alloca i64

; See that we only copy the x as y was not modified prior to coro.begin.
; CHECK:       store ptr @f.destroy, ptr %destroy.addr
; The next 3 instructions are to copy data in %x.addr from stack to frame.
; CHECK-NEXT:  %0 = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 3
; CHECK-NEXT:  %1 = load i64, ptr %x.addr, align 4
; CHECK-NEXT:  store i64 %1, ptr %0, align 4
; The next 3 instructions are to copy data in %z.addr from stack to frame.
; CHECK-NEXT:  [[T2:%.+]] = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 5
; CHECK-NEXT:  [[T3:%.+]] = load i64, ptr %z.addr, align 4
; CHECK-NEXT:  store i64 [[T3]], ptr [[T2]], align 4
; The next instruction is to recreate %y.cast in the original IR.
; CHECK-NEXT:  %y.addr.reload.addr = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 4
; CHECK-NEXT:  call void @llvm.memset.p0.i32(ptr %y.addr.reload.addr, i8 1, i32 4, i1 false)
; CHECK-NEXT:  %index.addr1 = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 6
; CHECK-NEXT:  store i1 false, ptr %index.addr1, align 1
; CHECK-NEXT:  ret ptr %hdl


declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1, token)

declare void @llvm.memset.p0.i32(ptr, i8, i32, i1)

declare noalias ptr @myAlloc(i32)
declare void @use(ptr)
declare void @free(ptr)
declare i1 @check()
