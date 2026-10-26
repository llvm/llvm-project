; RUN: opt -verify-each -passes='cgscc(function-attrs),ipsccp,cgscc(coro-split),instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='cgscc(coro-split),coro-cleanup,instcombine' -S %s | FileCheck %s

; A ThinLTO import can retain a declaration of the anonymous body rather than
; its definition. Its defining module will supply the split ramp. Do not leave
; the prepare intrinsic behind or transfer body-only return attributes to calls.

; CHECK-LABEL: declare ptr @external_body(ptr)
declare noalias nonnull ptr @external_body(ptr) noreturn presplitcoroutine "coro-retcon-body"
declare ptr @llvm.coro.prepare.retcon(ptr)
; A leftover coroutine declaration makes the cleanup pass run in this module.
declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr, ptr)

; CHECK-LABEL: define ptr @caller(
; CHECK: [[RESULT:%.*]] = call ptr @external_body(ptr %buffer)
; CHECK-NEXT: ret ptr [[RESULT]]
define ptr @caller(ptr %buffer) {
  %prepared = call ptr @llvm.coro.prepare.retcon(ptr @external_body)
  %value = call ptr %prepared(ptr %buffer)
  ret ptr %value
}
