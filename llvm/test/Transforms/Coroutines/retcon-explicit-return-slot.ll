; RUN: opt -verify-each -passes='ipsccp' -S %s | FileCheck %s --check-prefix=PRE
; RUN: opt -verify-each -passes='coro-early,ipsccp,coro-split' -S %s | FileCheck %s --check-prefix=SPLIT

; Both returned-continuation ABIs have an explicit pre-split return. The
; return slot is not part of the coroutine frame, even though it escapes via
; coro.id.retcon, and splitting replaces the placeholder load.

declare token @llvm.coro.id.retcon(i32, i32, ptr, ptr, ptr, ptr, ptr)
declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
declare ptr @malloc(i64)
declare void @free(ptr)
declare ptr @normal.prototype(ptr, i1)
declare void @once.prototype(ptr, i1)

define ptr @normal(ptr %buffer) presplitcoroutine {
entry:
  %return.slot = alloca ptr
  %id = call token @llvm.coro.id.retcon(i32 8, i32 8, ptr %buffer,
      ptr @normal.prototype, ptr @malloc, ptr @free, ptr %return.slot)
  %frame = call ptr @llvm.coro.begin(token %id, ptr null)
  %suspended = call i1 (...) @llvm.coro.suspend.retcon.i1()
  call void @llvm.coro.end(ptr %frame, i1 false, token none)
  %result = load ptr, ptr %return.slot
  ret ptr %result
}

define ptr @once(ptr %buffer) presplitcoroutine {
entry:
  %return.slot = alloca ptr
  %id = call token @llvm.coro.id.retcon.once(i32 8, i32 8, ptr %buffer,
      ptr @once.prototype, ptr @malloc, ptr @free, ptr %return.slot)
  %frame = call ptr @llvm.coro.begin(token %id, ptr null)
  %suspended = call i1 (...) @llvm.coro.suspend.retcon.i1()
  call void @llvm.coro.end(ptr %frame, i1 false, token none)
  %result = load ptr, ptr %return.slot
  ret ptr %result
}

; PRE-LABEL: define ptr @normal(
; PRE: %return.slot = alloca ptr
; PRE: call token @llvm.coro.id.retcon({{.*}}ptr %return.slot)
; PRE: load ptr, ptr %return.slot
; PRE: ret ptr
; PRE-LABEL: define ptr @once(
; PRE: %return.slot = alloca ptr
; PRE: call token @llvm.coro.id.retcon.once({{.*}}ptr %return.slot)
; PRE: load ptr, ptr %return.slot
; PRE: ret ptr

; SPLIT-LABEL: define ptr @normal(
; SPLIT: phi ptr [ @normal.resume.0
; SPLIT: ret ptr
; SPLIT-LABEL: define ptr @once(
; SPLIT: phi ptr [ @once.resume.0
; SPLIT: ret ptr
