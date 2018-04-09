; RUN: llc -mtriple=riscv32 -mattr=+c  -riscv-no-aliases -o %t1 < %s
; RUN: FileCheck %s < %t1

define void @foo() {
; CHECK-LABEL: foo:
; CHECK:   c.jr

end:
  ret void
}
