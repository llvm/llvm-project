; Need to move users of allocas that were moved into the coroutine frame after
; coro.begin.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define nonnull ptr @f(i32 %n) presplitcoroutine {
; CHECK-LABEL: @f(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ID:%.*]] = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr @f.resumers)
; CHECK-NEXT:    [[N_ADDR:%.*]] = alloca i32, align 4
; CHECK-NEXT:    store i32 [[N:%.*]], ptr [[N_ADDR]], align 4
; CHECK-NEXT:    [[CALL:%.*]] = tail call ptr @malloc(i32 24)
; CHECK-NEXT:    [[TMP0:%.*]] = tail call noalias nonnull ptr @llvm.coro.begin(token [[ID]], ptr [[CALL]])
; CHECK-NEXT:    store ptr @f.resume, ptr [[TMP0]], align 8
; CHECK-NEXT:    [[DESTROY_ADDR:%.*]] = getelementptr inbounds [[F_FRAME:%.*]], ptr [[TMP0]], i32 0, i32 1
; CHECK-NEXT:    store ptr @f.destroy, ptr [[DESTROY_ADDR]], align 8
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds [[F_FRAME]], ptr [[TMP0]], i32 0, i32 2
; CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[N_ADDR]], align 4
; CHECK-NEXT:    store i32 [[TMP2]], ptr [[TMP1]], align 4
;
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null);
  %n.addr = alloca i32
  store i32 %n, ptr %n.addr ; this needs to go after coro.begin
  %0 = tail call i32 @llvm.coro.size.i32()
  %call = tail call ptr @malloc(i32 %0)
  %1 = tail call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %call)
  call void @ctor(ptr %n.addr)
  br label %for.cond

for.cond:
  %2 = load i32, ptr %n.addr
  %dec = add nsw i32 %2, -1
  store i32 %dec, ptr %n.addr
  call void @print(i32 %2)
  %3 = call i8 @llvm.coro.suspend(token none, i1 false)
  %conv = sext i8 %3 to i32
  switch i32 %conv, label %coro_Suspend [
  i32 0, label %for.cond
  i32 1, label %coro_Cleanup
  ]

coro_Cleanup:
  %4 = call ptr @llvm.coro.free(token %id, ptr nonnull %1)
  call void @free(ptr %4)
  br label %coro_Suspend

coro_Suspend:
  call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret ptr %1
}

; CHECK-LABEL: @main
define i32 @main() {
entry:
  %hdl = call ptr @f(i32 4)
  call void @llvm.coro.resume(ptr %hdl)
  call void @llvm.coro.resume(ptr %hdl)
  call void @llvm.coro.destroy(ptr %hdl)
  ret i32 0
}

declare ptr @malloc(i32)
declare void @free(ptr)
declare void @print(i32)
declare void @ctor(ptr nocapture readonly)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i32 @llvm.coro.size.i32()
declare ptr @llvm.coro.begin(token, ptr)
declare i8 @llvm.coro.suspend(token, i1)
declare ptr @llvm.coro.free(token, ptr)
declare i1 @llvm.coro.end(ptr, i1, token)

declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)
