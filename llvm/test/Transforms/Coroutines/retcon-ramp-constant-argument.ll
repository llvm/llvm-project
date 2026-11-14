; RUN: opt -verify-each -passes='coro-early,ipsccp' -S %s | FileCheck %s --check-prefix=PRE
; RUN: opt -verify-each -passes='coro-early,ipsccp,cgscc(coro-split),coro-cleanup,ipsccp,instcombine' -S %s | FileCheck %s --check-prefix=POST
; RUN: opt -verify-each -passes='default<O2>,ipsccp,instcombine' -S %s | FileCheck %s --check-prefix=POST
; RUN: opt -verify-each -passes='thinlto-pre-link<O2>,thinlto<O2>,ipsccp,instcombine' -S %s | FileCheck %s --check-prefix=POST
;
; Before splitting, the opaque entry protects the caller's result. Once the
; ramp is exposed, ordinary IPO must recover propagation of a constant argument
; through the yielded value. This is an optimization-recovery check, not a
; measurement of the compile-time, code-size or runtime cost of outlining.

define internal { ptr, i32 } @yield_argument(ptr %buffer, i32 %value) noinline presplitcoroutine {
entry:
  %id = call token @llvm.coro.id.retcon.once(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free)
  %handle = call ptr @llvm.coro.begin(token %id, ptr null)
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 %value)
  call void @llvm.trap()
  unreachable

cleanup:
  call void @llvm.coro.end(ptr %handle, i1 false, token none)
  unreachable
}

; PRE-LABEL: define i32 @caller(
; PRE: [[PAIR:%.*]] = call { ptr, i32 } @yield_argument(ptr %buffer, i32 37)
; PRE-NEXT: [[VALUE:%.*]] = extractvalue { ptr, i32 } [[PAIR]], 1
; PRE-NEXT: ret i32 [[VALUE]]
; POST-LABEL: define{{.*}} i32 @caller(
; POST: ret i32 37
define i32 @caller(ptr %buffer) {
  %pair = call { ptr, i32 } @yield_argument(ptr %buffer, i32 37)
  %value = extractvalue { ptr, i32 } %pair, 1
  ret i32 %value
}

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
declare void @llvm.trap()
declare void @prototype(ptr, i1)
declare ptr @malloc(i64)
declare void @free(ptr)
