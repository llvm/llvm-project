; RUN: llc -mtriple=aarch64-- -fast-isel -fast-isel-abort=4 -verify-machineinstrs < %s | FileCheck %s

; Check that we ignore the assume intrinsic.

; CHECK-LABEL: test:
; CHECK: // %bb.0:
; CHECK-NEXT: ret
define void @test(i32 %a) {
  %tmp0 = icmp slt i32 %a, 0
  call void @llvm.assume(i1 %tmp0)
  ret void
}

declare void @llvm.assume(i1)
