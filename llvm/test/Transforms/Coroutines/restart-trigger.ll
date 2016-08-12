; Verifies that restart trigger forces IPO pipelines restart and the same
; coroutine is looked at by CoroSplit pass twice.
; REQUIRES: asserts
; RUN: opt < %s -S -O0 -enable-coroutines -debug-only=coro-split 2>&1 | FileCheck %s
; RUN: opt < %s -S -O1 -enable-coroutines -debug-only=coro-split 2>&1 | FileCheck %s

; CHECK:      CoroSplit: Processing coroutine 'f' state: 0
; CHECK-NEXT: CoroSplit: Processing coroutine 'f' state: 1

declare token @llvm.coro.id(i32, i8*, i8*)
declare i8* @llvm.coro.begin(token, i8*)

; a coroutine start function
define void @f() {
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null)
  call i8* @llvm.coro.begin(token %id, i8* null)
  ret void
}
