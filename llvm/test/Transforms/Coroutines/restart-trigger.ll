; REQUIRES: asserts
; The following tests use the new pass manager, and verify that the coroutine
; passes re-run the CGSCC pipeline.
; RUN: opt < %s -S -passes='default<O0>' -enable-coroutines -debug-only=coro-split 2>&1 | FileCheck --check-prefix=CHECK-NEWPM %s
; RUN: opt < %s -S -passes='default<O1>' -enable-coroutines -debug-only=coro-split 2>&1 | FileCheck --check-prefix=CHECK-NEWPM %s

; CHECK:      CoroSplit: Processing coroutine 'f' state: 0
; CHECK-NEXT: CoroSplit: Processing coroutine 'f' state: 1
; CHECK-NEWPM:      CoroSplit: Processing coroutine 'f' state: 0
; CHECK-NEWPM-NOT:  CoroSplit: Processing coroutine 'f' state: 1


define void @f() {
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
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
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret void  
}

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i8* @llvm.coro.free(token, i8*)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)
declare i1 @llvm.coro.end(i8*, i1) 

declare noalias i8* @malloc(i32)
declare void @print(i32)
declare void @free(i8*)
