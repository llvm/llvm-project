; RUN: llc < %s -emit-gnuas-syntax-on-zos=0 --mtriple=s390x-ibm-zos | FileCheck --match-full-lines %s

; empty function
; CHECK: foo
; CHECK-NEXT: L#func_end0
define void @foo() {
entry:
  ret void
}
