; RUN: llc < %s -mtriple=arm64-apple-darwin | FileCheck %s

define void @test_shadow_optimization() {
entry:
; Expect 8 bytes worth of nops here rather than 16: With the shadow optimization
; in place, 8 bytes will be consumed by the frame teardown and return instr.
; CHECK-LABEL: test_shadow_optimization:
; CHECK:      nop
; CHECK-NEXT: nop
; CHECK-NOT:  nop
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  0, i32  16)
  ret void
}

declare void @llvm.experimental.stackmap(i64, i32, ...)
