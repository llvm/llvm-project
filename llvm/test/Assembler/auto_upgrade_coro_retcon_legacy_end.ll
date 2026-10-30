; RUN: opt -S %s | FileCheck %s

; The old coro.end declaration may already have been renamed to .old by its
; own auto-upgrade when the retcon intrinsic is upgraded. The return must
; become explicit regardless of the order in which declarations are visited.

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end.old(ptr, i1, token)
declare void @resume(ptr, i1)
declare ptr @malloc(i64)
declare void @free(ptr)

define ptr @legacy_end(ptr %buffer) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id.retcon.once(i32 8, i32 8, ptr %buffer,
      ptr @resume, ptr @malloc, ptr @free)
  %frame = call ptr @llvm.coro.begin(token %id, ptr null)
  call void @llvm.coro.end.old(ptr %frame, i1 false, token none)
  unreachable
}

; CHECK-LABEL: define ptr @legacy_end(
; CHECK: %coro.ret = alloca ptr
; CHECK: call token @llvm.coro.id.retcon.once({{.*}}ptr %coro.ret)
; CHECK: call void @llvm.coro.end.old(ptr %frame, i1 false, token none)
; CHECK: load ptr, ptr %coro.ret
; CHECK: ret ptr
