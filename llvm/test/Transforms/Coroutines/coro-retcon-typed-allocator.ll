; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @typed_allocator(ptr noalias dereferenceable(32) %buffer) presplitcoroutine {
; CHECK-LABEL: @typed_allocator(
; CHECK: call ptr @swift_coroFrameAlloc(i64 40, i64 123)
entry:
  %value = alloca [5 x i64], align 8
  %id = call token (i32, i32, ptr, ptr, ptr, ptr, ...) @llvm.coro.id.retcon.once(i32 32, i32 8, ptr %buffer, ptr @prototype, ptr @swift_coroFrameAlloc, ptr @free, i64 123)
  %handle = call ptr @llvm.coro.begin(token %id, ptr null)
  call void @use(ptr %value)
  %suspend = call i1 (...) @llvm.coro.suspend.retcon.i1()
  br i1 %suspend, label %cleanup, label %resume

resume:
  call void @use(ptr %value)
  br label %cleanup

cleanup:
  call void @llvm.coro.end(ptr %handle, i1 false, token none)
  unreachable
}

declare void @prototype(ptr, i1 zeroext)
declare ptr @swift_coroFrameAlloc(i64, i64)
declare void @free(ptr)
declare void @use(ptr)
declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr, ...)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
