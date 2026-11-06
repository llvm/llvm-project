; RUN: llc -mtriple=riscv32 -mattr=+zbb -global-isel -global-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -O0 -mtriple=riscv32 -mattr=+zbb -global-isel -global-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

declare i32 @llvm.riscv.orc.b.i32(i32)

define i32 @orc_b_i32(i32 %x) {
; CHECK-LABEL: orc_b_i32:
; CHECK: orc.b a0, a0
; CHECK-NEXT: ret
  %r = call i32 @llvm.riscv.orc.b.i32(i32 %x)
  ret i32 %r
}
