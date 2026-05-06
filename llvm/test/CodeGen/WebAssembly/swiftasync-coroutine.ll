; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split),module(coro-cleanup)' -S -mtriple=wasm32-unknown-unknown -mattr=+tail-call | FileCheck --check-prefix=IR %s
; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split),module(coro-cleanup)' -mtriple=wasm32-unknown-unknown -mattr=+tail-call | \
; RUN:   llc -mtriple=wasm32-unknown-unknown -verify-machineinstrs \
; RUN:   -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals \
; RUN:   -wasm-keep-registers -mattr=+tail-call | FileCheck --check-prefix=ASM %s

; End-to-end test: verify that async coroutine splitting with swifttailcc
; produces musttail calls (at the IR level) that lower to return_call (at the
; assembly level) when the tail-call feature is enabled.

%swift.async_func_pointer = type <{ i32, i32 }>
@checkTu = global %swift.async_func_pointer <{ i32 ptrtoint (ptr @check to i32), i32 8 }>

define swifttailcc void @check(ptr swiftasync %0) {
; IR-LABEL: define swifttailcc void @check(
; IR:         musttail call swifttailcc void @check.0()
;
; Coroutine splitting generates a resume function with swifttailcc and swiftasync.
; IR-LABEL: define internal swifttailcc void @checkTQ0_(ptr swiftasync
;
; ASM-LABEL: check:
; ASM:         return_call check.0
entry:
  %1 = call token @llvm.coro.id.async(i32 0, i32 0, i32 0, ptr @checkTu)
  %2 = call ptr @llvm.coro.begin(token %1, ptr null)
  %3 = call ptr @llvm.coro.async.resume()
  store ptr %3, ptr %0, align 4
  %4 = call { ptr, i32 } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0i32s(i32 0, ptr %3, ptr @__swift_async_resume_project_context, ptr @check.0, ptr null, ptr null)
  ret void
}

declare swifttailcc void @check.0()
declare { ptr, i32 } @llvm.coro.suspend.async.sl_p0i32s(i32, ptr, ptr, ...)
declare token @llvm.coro.id.async(i32, i32, i32, ptr)
declare ptr @llvm.coro.begin(token, ptr writeonly)
declare ptr @llvm.coro.async.resume()

define ptr @__swift_async_resume_project_context(ptr %0) {
entry:
  ret ptr null
}
