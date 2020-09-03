; RUN: llc -verify-machineinstrs -O3 -mcpu=skylake -x86-align-branch-boundary=32 -x86-align-branch=call -filetype=obj < %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

;; This file is a companion to align-branch-boundary-suppressions.ll.
;; It exists to demonstrate that suppressions are actually wired into the
;; integrated assembler.

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define void @test_statepoint(i32 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK: 1: callq
; CHECK-NEXT: 6: callq
; CHECK-NEXT: b: callq
; CHECK-NEXT: 10: callq
; CHECK-NEXT: 15: callq
; CHECK-NEXT: 1a: callq
; CHECK-NEXT: 1f: callq
entry:
  ; Each of these will be 5 bytes, pushing the statepoint to offset=30.
  ; For a normal call, this would force padding between the last normal
  ; call and the safepoint, but since we've suppressed alignment that won't
  ; happen for the safepoint.  That's non-ideal, we'd really prefer to do
  ; the alignment and just keep the label with the statepoint call. (TODO)
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0)
  ret void
}

declare void @foo()
declare zeroext i1 @return_i1()
declare token @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, i1 ()*, i32, i32, ...)
