; RUN: llc -mtriple=riscv64 -mattr=+zbb -global-isel -global-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -O0 -mtriple=riscv64 -mattr=+zbb -global-isel -global-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

declare i64 @llvm.riscv.orc.b.i64(i64)
declare i32 @llvm.riscv.orc.b.i32(i32)

define i64 @orc_b_i64(i64 %x) {
; CHECK-LABEL: orc_b_i64:
; CHECK: orc.b a0, a0
; CHECK-NEXT: ret
  %r = call i64 @llvm.riscv.orc.b.i64(i64 %x)
  ret i64 %r
}

define signext i32 @orc_b_i32(i32 signext %x) {
; CHECK-LABEL: orc_b_i32:
; CHECK: orc.b a0, a0
; CHECK-NEXT: sext.w a0, a0
; CHECK-NEXT: ret
  %r = call i32 @llvm.riscv.orc.b.i32(i32 %x)
  ret i32 %r
}

define signext i32 @orc_b_i32_zext(i32 zeroext %x) {
; CHECK-LABEL: orc_b_i32_zext:
; CHECK: orc.b a0, a0
; CHECK-NEXT: sext.w a0, a0
; CHECK-NEXT: ret
  %r = call i32 @llvm.riscv.orc.b.i32(i32 %x)
  ret i32 %r
}

define i32 @orc_b_i32_anyext(i32 %x) {
; CHECK-LABEL: orc_b_i32_anyext:
; CHECK: orc.b a0, a0
; CHECK-NEXT: ret
  %r = call i32 @llvm.riscv.orc.b.i32(i32 %x)
  ret i32 %r
}
