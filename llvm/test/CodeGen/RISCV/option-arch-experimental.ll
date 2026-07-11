; RUN: llc -mtriple=riscv64 < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 < %s \
; RUN:   | llvm-mc -triple=riscv64 -mattr=+experimental -filetype=obj -o /dev/null

; CHECK: .option push
; CHECK-NEXT: .option arch, +zicfiss, +zicsr, +zimop
; CHECK-NOT: experimental-
define void @f() "target-features"="+experimental-zicfiss" {
; CHECK-LABEL: f:
; CHECK: .option pop
entry:
  ret void
}
