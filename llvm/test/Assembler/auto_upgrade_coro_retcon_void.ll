; RUN: opt -S %s | FileCheck %s
; RUN: opt -passes=function-attrs -S %s | FileCheck %s

; Void coroutines need an explicit ret but no return-value alloca.
define void @legacy_void() presplitcoroutine {
  call token @llvm.coro.id.retcon.once(i32 0, i32 0, ptr null, ptr @legacy_void, ptr null, ptr null)
  call void (ptr, i1, ...) @llvm.coro.end(ptr null, i1 false)
  unreachable
}

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr)
declare void @llvm.coro.end(ptr, i1, ...)

; CHECK-NOT: noreturn
; CHECK-LABEL: define void @legacy_void(
; CHECK-NOT: alloca
; CHECK: call token @llvm.coro.id.retcon.once({{.*}}ptr null)
; CHECK: ret void
; CHECK-NOT: noreturn
