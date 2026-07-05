; RUN: llc --mtriple=loongarch64 --verify-machineinstrs --stop-after=prologepilog < %s | FileCheck %s

;; Check that STATEPOINT instruction has an early clobber implicit def for R1.

define void @test() gc "statepoint-example" {
entry:
  %safepoint_token = tail call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live" ()]
; CHECK: STATEPOINT 0, 0, 0, target-flags(loongarch-call-plt) @return_i1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, csr_ilp32d_lp64d, implicit-def $r3, implicit-def dead early-clobber $r1
  ret void
}

declare void @return_i1()
declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
