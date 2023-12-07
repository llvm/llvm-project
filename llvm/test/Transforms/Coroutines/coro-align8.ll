; Tests that the coro.align intrinsic could be lowered to correct alignment
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @f() presplitcoroutine {
entry:
  %x = alloca i64
  %y = alloca i64
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %align = call i32 @llvm.coro.align.i32()
  %alloc = call ptr @aligned_alloc(i32 %align, i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  call void @capture_call(ptr %x)
  call void @nocapture_call(ptr %y)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0)
  ret ptr %hdl
}

; %x needs to go to the frame since it's escaped; %y will stay as local since it doesn't escape.
; CHECK:        %f.Frame = type { ptr, ptr, i64, i1 }
; CHECK-LABEL:  define ptr @f()
; CHECK:          %[[ALLOC:.+]] = call ptr @aligned_alloc(i32 8, i32 32)
; CHECK-NEXT:     call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %[[ALLOC]])

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i32 @llvm.coro.align.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1)

declare void @capture_call(ptr)
declare void @nocapture_call(ptr nocapture)
declare noalias ptr @aligned_alloc(i32, i32)
declare void @free(ptr)
