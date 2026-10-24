; RUN: opt -verify-each -passes='coro-early,coro-split' -S %s | FileCheck %s
; RUN: sed 's/alloca i8, i64 16, align 8/alloca i8, i64 32, align 8/' %s | opt -verify-each -passes='coro-early,coro-split' -S | FileCheck %s
; RUN: sed 's/alloca i8, i64 16, align 8/alloca i8, i64 8, align 8/' %s | not --crash opt -passes='coro-early,coro-split' -disable-output 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: sed 's/alloca i8, i64 16, align 8/alloca i8, i64 16, align 1/' %s | not --crash opt -passes='coro-early,coro-split' -disable-output 2>&1 | FileCheck %s --check-prefix=INVALID

; Byte-backed storage is valid if its size and alignment suffice for the
; aggregate return. Undersized and underaligned storage must be rejected.

target datalayout = "e-p:64:64"

define { ptr, ptr } @f(ptr %buffer, ptr %value) presplitcoroutine {
entry:
  %return.slot = alloca i8, i64 16, align 8
  %id = call token @llvm.coro.id.retcon.once(i32 8, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free, ptr %return.slot)
  %frame = call ptr @llvm.coro.begin(token %id, ptr null)
  %suspended = call i1 (...) @llvm.coro.suspend.retcon.i1(ptr %value)
  call void @llvm.coro.end(ptr %frame, i1 false, token none)
  %result = load { ptr, ptr }, ptr %return.slot, align 8
  ret { ptr, ptr } %result
}

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
declare void @prototype(ptr, i1)
declare ptr @malloc(i64)
declare void @free(ptr)

; CHECK-LABEL: define { ptr, ptr } @f(
; CHECK: ret { ptr, ptr }
; CHECK-LABEL: define internal void @f.resume.0(
; CHECK: ret void
; INVALID: return slot of coro.id.retcon.* has insufficient size or alignment
