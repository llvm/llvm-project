; RUN: llc -filetype=asm -o - -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: Lxray_sled_0:
; CHECK-NEXT:  .p2align 2, 0x90
; CHECK-NEXT:  .ascii "\353\t"
; CHECK-NEXT:  nopw 512(%rax,%rax)
; CHECK-LABEL: Ltmp0:
  ret i32 0
; CHECK-LABEL: Lxray_sled_1:
; CHECK-NEXT:  retq
; CHECK-NEXT:  nopw %cs:512(%rax,%rax)
}
