; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-gnu-linux | FileCheck %s

define void @test_shadow_optimization() {
entry:
; Expect 12 bytes worth of nops here rather than 32: With the shadow optimization
; in place, 20 bytes will be consumed by the frame teardown and return instr.
; CHECK-LABEL: test_shadow_optimization:

; CHECK:      nop
; CHECK-NEXT: nop
; CHECK-NEXT: nop
; CHECK-NOT:  nop
; CHECK: addi 1, 1, 64
; CHECK: ld [[REG1:[0-9]+]], 16(1)
; CHECK: ld 31, -8(1)
; CHECK: mtlr [[REG1]]
; CHECK: blr

  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  0, i32  32)
  ret void
}

declare void @llvm.experimental.stackmap(i64, i32, ...)

