; Verifies that restart trigger forces IPO pipelines restart and the same
; coroutine is looked at by CoroSplit pass twice.
; REQUIRES: asserts
; RUN: opt < %s -S -O0 -enable-coroutines -debug-only=coro-split 2>&1 | FileCheck %s
; RUN: opt < %s -S -O1 -enable-coroutines -debug-only=coro-split 2>&1 | FileCheck %s

; CHECK:      CoroSplit: Processing coroutine 'f' state: 0
; CHECK-NEXT: CoroSplit: Processing coroutine 'f' state: 1

define void @f() {
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  call void @print(i32 0)
  %s1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %s1, label %suspend [i8 0, label %resume 
                                 i8 1, label %cleanup]
resume:
  call void @print(i32 1)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(i8* %hdl, i1 0)
  ret void  
}

declare token @llvm.coro.id(i32, i8*, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i8* @llvm.coro.free(i8*)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)
declare void @llvm.coro.end(i8*, i1) 

declare noalias i8* @malloc(i32)
declare void @print(i32)
declare void @free(i8*)
