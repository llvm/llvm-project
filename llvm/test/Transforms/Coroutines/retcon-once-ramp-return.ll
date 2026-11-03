; RUN: opt -verify-each -passes='coro-early,cgscc(coro-split),ipsccp,coro-cleanup,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='coro-early,ipsccp,cgscc(coro-split),coro-cleanup,ipsccp,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='default<O2>,ipsccp,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='thinlto-pre-link<O2>,thinlto<O2>,ipsccp,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='thinlto<O2>,ipsccp,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='lto-pre-link<O2>,lto<O2>,ipsccp,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='lto<O0>,ipsccp,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='lto<O1>,ipsccp,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='lto<O2>,ipsccp,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='lto<O3>,ipsccp,instcombine' -S %s | FileCheck %s

; The body never returns, but the ramp returns the yielded value at the first
; suspension. Body exits do not describe that caller-visible return.
; Splitting before IPSCCP preserves the value;
; running IPO before splitting must preserve it too.
; A final IPSCCP pass folds the now-visible ramp return, but cannot repair an
; earlier incorrect replacement of the caller's result with undef.
; Older producers did not separate the entry in their CoroEarly pass. Their
; bitcode must also work when sent directly to either LTO backend.

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
declare void @llvm.trap()
declare void @prototype(ptr, i1)
declare ptr @malloc(i64)
declare void @free(ptr)

define internal { ptr, i32 } @yield_once(ptr noalias %buffer) noinline presplitcoroutine {
entry:
  %id = call token @llvm.coro.id.retcon.once(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7)
  call void @llvm.trap()
  unreachable

cleanup:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  unreachable
}

; CHECK-LABEL: define{{.*}} i32 @caller(
; CHECK: ret i32 7
define i32 @caller(ptr %buffer) {
  %pair = call { ptr, i32 } @yield_once(ptr %buffer)
  %yield = extractvalue { ptr, i32 } %pair, 1
  ret i32 %yield
}
