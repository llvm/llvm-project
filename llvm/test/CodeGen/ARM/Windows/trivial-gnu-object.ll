; RUN: llc -mtriple=thumbv7-windows-itanium -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
; RUN: llc -mtriple=thumbv7-windows-gnu -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s

define void @foo() {
; CHECK: file format coff-arm

; CHECK-LABEL: <foo>:
; CHECK: bx lr
  ret void
}
