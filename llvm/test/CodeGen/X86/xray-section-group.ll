; RUN: llc -verify-machineinstrs -filetype=asm -o - -mtriple=x86_64-unknown-linux-gnu -function-sections < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -filetype=obj -o %t -mtriple=x86_64-unknown-linux-gnu -function-sections < %s
; RUN: llvm-objdump -triple x86_64-unknown-linux-gnu -disassemble-all %t | FileCheck %s --check-prefix=CHECK-OBJ

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK: .section .text.foo,"ax",@progbits
  ret i32 0
; CHECK: .section xray_instr_map,"awo",@progbits,foo,unique,1
}

$bar = comdat any
define i32 @bar() nounwind noinline uwtable "function-instrument"="xray-always" comdat($bar) {
; CHECK: .section .text.bar,"axG",@progbits,bar,comdat
  ret i32 1
; CHECK: .section xray_instr_map,"aGwo",@progbits,bar,comdat,bar,unique,2
}

; CHECK-OBJ:      section xray_instr_map:
