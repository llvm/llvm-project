; RUN: opt -verify-each -passes='coro-early,cgscc(coro-split),ipsccp,coro-cleanup,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='coro-early,ipsccp,cgscc(coro-split),coro-cleanup,ipsccp,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='default<O2>,ipsccp,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='thinlto-pre-link<O2>,thinlto<O2>,ipsccp,instcombine' -S %s | FileCheck %s

; The body never returns, but the ramp returns the yielded value at the first
; suspension. An explicit return after the unreachable coro.end does not model
; that caller-visible return. Splitting before IPSCCP preserves the value;
; running IPO before splitting must preserve it too.
; A final IPSCCP pass folds the now-visible ramp return, but cannot repair an
; earlier incorrect replacement of the caller's result with undef.
; The ThinLTO backend expects the pre-link pipeline to have run CoroEarly.

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
declare void @llvm.trap()
declare void @prototype(ptr, i1)
declare ptr @malloc(i64)
declare void @free(ptr)

define internal { ptr, i32 } @yield_once(ptr noalias %buffer) noinline presplitcoroutine {
entry:
  %return.slot = alloca { ptr, i32 }, align 8
  %id = call token @llvm.coro.id.retcon.once(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free, ptr %return.slot)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7)
  call void @llvm.trap()
  unreachable

cleanup:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  %result = load { ptr, i32 }, ptr %return.slot, align 8
  ret { ptr, i32 } %result
}

; CHECK-LABEL: define{{.*}} i32 @caller(
; CHECK: ret i32 7
define i32 @caller(ptr %buffer) {
  %pair = call { ptr, i32 } @yield_once(ptr %buffer)
  %yield = extractvalue { ptr, i32 } %pair, 1
  ret i32 %yield
}
