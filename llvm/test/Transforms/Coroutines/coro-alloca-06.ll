; Test that in some simple cases allocas will not live on the frame even
; though their pointers are stored.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

%handle = type { ptr }

define ptr @f() presplitcoroutine {
entry:
  %0 = alloca %"handle", align 8
  %1 = alloca ptr, align 8
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  br label %tricky

tricky:
  %2 = call ptr @await_suspend()
  store ptr %2, ptr %0, align 8
  call void @llvm.lifetime.start.p0(ptr %1)
  store ptr %0, ptr %1, align 8
  %3 = load ptr, ptr %1, align 8
  %4 = load ptr, ptr %3, align 8
  call void @llvm.lifetime.end.p0(ptr %1)
  br label %finish

finish:
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
  i8 1, label %cleanup]
resume:
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; CHECK:        %f.Frame = type { ptr, ptr, i1 }
; CHECK-LABEL: @f(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = alloca [[HANDLE:%.*]], align 8
; CHECK-NEXT:    [[TMP1:%.*]] = alloca ptr, align 8

; CHECK:         [[TMP2:%.*]] = call ptr @await_suspend()
; CHECK-NEXT:    store ptr [[TMP2]], ptr [[TMP0]], align 8
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(ptr [[TMP1]])
; CHECK-NEXT:    store ptr [[TMP0]], ptr [[TMP1]], align 8
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(ptr [[TMP1]])
;

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1, token)

declare void @llvm.lifetime.start.p0(ptr nocapture)
declare void @llvm.lifetime.end.p0(ptr nocapture)

declare ptr @await_suspend()
declare void @print(ptr nocapture)
declare noalias ptr @malloc(i32)
declare void @free(ptr)
