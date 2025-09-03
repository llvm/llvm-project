; RUN: llc -mtriple=riscv64 -global-isel -O0 < %s | FileCheck %s

target triple = "riscv64"

define i32 @icmp_ult_imm(i32 %x) {
; CHECK-LABEL: icmp_ult_imm:
; CHECK:       sltiu
; CHECK:       ret
entry:
  %c = icmp ult i32 %x, 15
  %z = zext i1 %c to i32
  ret i32 %z
}
