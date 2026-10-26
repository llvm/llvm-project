; RUN: opt -verify-each -passes='coro-early' -S %s | FileCheck %s --check-prefix=EARLY
; RUN: opt -verify-each -passes='coro-early,coro-early' -S %s | FileCheck %s --check-prefix=EARLY
; RUN: opt -verify-each -passes='coro-early,cgscc(function-attrs),ipsccp' -S %s | FileCheck %s --check-prefix=IPO
; RUN: opt -verify-each -passes='coro-early,ipsccp,cgscc(coro-split),coro-cleanup,ipsccp,instcombine' -S %s | FileCheck %s --check-prefix=POST
; RUN: opt -verify-each -passes='default<O2>,ipsccp,instcombine' -S %s | FileCheck %s --check-prefix=POST
; RUN: opt -verify-each -passes='thinlto-pre-link<O2>,thinlto<O2>,ipsccp,instcombine' -S %s | FileCheck %s --check-prefix=POST

; A global and an alias must still name the caller-visible entry, not its body.
; EARLY: @published = global ptr @yield_many
; EARLY: @alias = alias { ptr, i32 } (ptr), ptr @yield_many
@published = global ptr @yield_many
@alias = alias { ptr, i32 } (ptr), ptr @yield_many

declare token @llvm.coro.id.retcon(i32, i32, ptr, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
declare void @llvm.trap()
declare { ptr, i32 } @prototype(ptr, i1)
declare ptr @malloc(i64)
declare void @free(ptr)

; The body yields repeatedly and never exits. Abnormal resumption traps rather
; than suspending again. Its caller-visible ramp must nevertheless return 7.
; EARLY-LABEL: define fastcc { ptr, i32 } @yield_many(
; EARLY: [[PREPARED:%.*]] = call ptr @llvm.coro.prepare.retcon(ptr @yield_many.coro.body)
; EARLY-NEXT: [[RESULT:%.*]] = musttail call fastcc { ptr, i32 } [[PREPARED]](ptr %buffer)
; EARLY-NEXT: ret { ptr, i32 } [[RESULT]]
; EARLY-LABEL: define internal fastcc { ptr, i32 } @yield_many.coro.body(
; EARLY: call token @llvm.coro.id.retcon
; EARLY: call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7)
define fastcc { ptr, i32 } @yield_many(ptr %buffer) noinline presplitcoroutine {
entry:
  %return.slot = alloca { ptr, i32 }, align 8
  %id = call token @llvm.coro.id.retcon(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free, ptr %return.slot)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  br label %again

again:
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7)
  br i1 %unwind, label %abnormal, label %again

abnormal:
  call void @llvm.trap()
  unreachable

cleanup:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  %result = load { ptr, i32 }, ptr %return.slot, align 8
  ret { ptr, i32 } %result
}

; IPO-LABEL: define i32 @caller(
; IPO: [[PAIR:%.*]] = call fastcc { ptr, i32 } @yield_many(ptr %buffer)
; IPO: [[YIELD:%.*]] = extractvalue { ptr, i32 } [[PAIR]], 1
; IPO: ret i32 [[YIELD]]
; POST-LABEL: define{{.*}} i32 @caller(
; POST: ret i32 7
define i32 @caller(ptr %buffer) {
  %pair = call fastcc { ptr, i32 } @yield_many(ptr %buffer)
  %yield = extractvalue { ptr, i32 } %pair, 1
  ret i32 %yield
}

; Ordinary functions do not acquire wrappers or lose return-value propagation.
; POST-LABEL: define{{.*}} i32 @ordinary_caller(
; POST: ret i32 9
define internal i32 @ordinary() {
  ret i32 9
}

define i32 @ordinary_caller() {
  %value = call i32 @ordinary()
  ret i32 %value
}
