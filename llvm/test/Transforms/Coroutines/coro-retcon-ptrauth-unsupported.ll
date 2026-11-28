; A signed prototype may only carry a schema that CoroSplit can reproduce when
; it signs the continuation functions. Check that the two schemas it cannot
; reproduce are rejected: a deactivation symbol, and an address discriminator
; other than the coroutine-storage marker.

; RUN: rm -rf %t && split-file %s %t
; RUN: not --crash opt -passes='module(coro-early),cgscc(coro-split)' -disable-output %t/deactivation-symbol.ll 2>&1 | FileCheck %s
; RUN: not --crash opt -passes='module(coro-early),cgscc(coro-split)' -disable-output %t/address-discriminator.ll 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: llvm.coro.id.retcon.* prototype has an unsupported schema

;--- deactivation-symbol.ll
declare void @prototype(ptr, i1)
declare noalias ptr @allocate(i32)
declare void @deallocate(ptr)

@ds = external global i8

define ptr @f(ptr %buffer) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id.retcon.once(i32 8, i32 8, ptr %buffer, ptr ptrauth (ptr @prototype, i32 0, i64 3909, ptr inttoptr (i64 1 to ptr), ptr @ds), ptr @allocate, ptr @deallocate)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1()
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  unreachable
}

;--- address-discriminator.ll
declare void @prototype(ptr, i1)
declare noalias ptr @allocate(i32)
declare void @deallocate(ptr)

@slot = external global ptr

define ptr @f(ptr %buffer) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id.retcon.once(i32 8, i32 8, ptr %buffer, ptr ptrauth (ptr @prototype, i32 0, i64 3909, ptr @slot), ptr @allocate, ptr @deallocate)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1()
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  unreachable
}
