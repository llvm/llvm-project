; RUN: llc --mtriple=riscv32 --mattr=+use-postra-scheduler --frame-pointer=all < %s | FileCheck %s
; RUN: llc --mtriple=riscv64 --mattr=+use-postra-scheduler --frame-pointer=all < %s | FileCheck %s

; CHECK-LABEL: foo:
; CHECK: addi	[[REG1:[a-z0-9]+]], [[REG1]], 1
; CHECK: sw	[[REG1]], {{[0-9]+}}({{[a-z0-9]+}})
; CHECK: {{lw|ld}}	s0, {{[0-9]+}}(sp)
define void @foo() nounwind sanitize_address {
entry:
  %1 = load ptr, ptr inttoptr (i32 64 to ptr), align 64
  %2 = load i32, ptr %1, align 8
  %3 = add nsw i32 %2, 1
  store i32 %3, ptr %1, align 8
  ret void
}
