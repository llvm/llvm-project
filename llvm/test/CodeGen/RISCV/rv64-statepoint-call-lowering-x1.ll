; RUN: llc -mtriple riscv64 -verify-machineinstrs -stop-after=prologepilog < %s | FileCheck %s

; Check that STATEPOINT instruction has an early clobber implicit def for LR.
target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "riscv64"

define void @test() "frame-pointer"="all" gc "statepoint-example" {
entry:
  %safepoint_token = tail call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live" ()]
; CHECK: STATEPOINT 0, 0, 0, target-flags(riscv-call) @return_i1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, csr_ilp32_lp64, implicit-def $x2, implicit-def dead early-clobber $x1
  ret void
}


declare void @return_i1()
declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
