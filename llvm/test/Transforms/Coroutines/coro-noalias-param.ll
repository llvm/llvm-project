; RUN: opt < %s -S -passes=coro-early | FileCheck %s
%struct.A = type <{ i64, i64, i32, [4 x i8] }>

define void @f(ptr nocapture readonly noalias align 8 %a) presplitcoroutine {
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  call void @print(i32 0)
  %s1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %s1, label %suspend [i8 0, label %resume 
                                 i8 1, label %cleanup]
resume:
  call void @print(i32 1)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret void  
}

; check that the noalias attribute is removed from the argument
; CHECK: define void @f(ptr nocapture readonly align 8 %a)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)
declare i1 @llvm.coro.end(ptr, i1, token) 

declare noalias ptr @malloc(i32)
declare void @print(i32)
declare void @free(ptr)
