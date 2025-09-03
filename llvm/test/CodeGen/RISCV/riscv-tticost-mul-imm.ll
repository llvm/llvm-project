; REQUIRES: riscv-registered-target
; RUN: opt -passes="print<cost-model>" -disable-output -mtriple=riscv64 %s 2>&1 | FileCheck %s

define i32 @mul_imm_cost(i32 %x) {
; CHECK-NOT: Found an estimated cost of 0 for instruction: %s = mul i32 %x, 123
; CHECK:     Found an estimated cost of {{[1-9][0-9]*}} for instruction: %s = mul i32 %x, 123
  %s = mul i32 %x, 123
  ret i32 %s
}
