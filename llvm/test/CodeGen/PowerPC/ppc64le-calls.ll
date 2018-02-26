; RUN: llc -verify-machineinstrs -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs < %s | FileCheck %s

; The second run of the test case is to ensure the behaviour is the same
; without specifying -mcpu=pwr8 as that is now the baseline for ppc64le.

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Indirect calls requires a full stub creation
define void @test_indirect(void ()* nocapture %fp) {
; CHECK-LABEL: @test_indirect
  tail call void %fp()
; CHECK-DAG: std 2, 24(1)
; CHECK-DAG: mr 12, 3
; CHECK-DAG: mtctr 3
; CHECK: bctrl
; CHECK-NEXT: ld 2, 24(1)
  ret void
}

