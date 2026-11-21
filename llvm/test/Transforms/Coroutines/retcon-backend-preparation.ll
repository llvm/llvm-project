; RUN: opt -verify-each -passes='thinlto<O2>,ipsccp,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='lto<O2>,ipsccp,instcombine' -S %s | FileCheck %s

; Preparing retcon input from an older producer must not replay frontend
; lowering on unrelated imported functions. In particular, the CFI import path
; can contain a switched-resume coro.id without the presplitcoroutine attribute.
; The retcon entry still needs its boundary before backend IPO.

define available_externally ptr @imported_switched() {
  %id = call token @llvm.coro.id(i32 8, ptr null, ptr @imported_switched, ptr null)
  %handle = call ptr @llvm.coro.begin(token %id, ptr null)
  ret ptr %handle
}

define internal { ptr, i32 } @yield_once(ptr %buffer) noinline presplitcoroutine {
entry:
  %id = call token @llvm.coro.id.retcon.once(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free)
  %handle = call ptr @llvm.coro.begin(token %id, ptr null)
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7)
  call void @llvm.trap()
  unreachable

cleanup:
  call void @llvm.coro.end(ptr %handle, i1 false, token none)
  unreachable
}

; CHECK-LABEL: define{{.*}} i32 @caller(
; CHECK: ret i32 7
define i32 @caller(ptr %buffer) {
  %pair = call { ptr, i32 } @yield_once(ptr %buffer)
  %yield = extractvalue { ptr, i32 } %pair, 1
  ret i32 %yield
}

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
declare void @llvm.trap()
declare void @prototype(ptr, i1)
declare ptr @malloc(i64)
declare void @free(ptr)
