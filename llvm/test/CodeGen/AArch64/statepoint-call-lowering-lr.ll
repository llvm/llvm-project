; RUN: llc -mtriple aarch64-none-linux-gnu -verify-machineinstrs -stop-after=prologepilog < %s | FileCheck %s

; Check that STATEPOINT instruction has an early clobber implicit def for LR.
target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define void @test() "frame-pointer"="all" gc "statepoint-example" {
entry:
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, void ()* elementtype(void ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live" ()]
; CHECK: STATEPOINT 0, 0, 0, @return_i1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, csr_aarch64_aapcs, implicit-def $sp, implicit-def dead early-clobber $lr
  ret void
}


declare void @return_i1()
declare token @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, void ()*, i32, i32, ...)
