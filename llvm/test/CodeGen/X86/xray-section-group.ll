; RUN: llc -filetype=asm -o - -mtriple=x86_64-unknown-linux-gnu -function-sections < %s | FileCheck %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK: .section .text.foo,"ax",@progbits
  ret i32 0
; CHECK: .section xray_instr_map,"a",@progbits
}

$bar = comdat any
define i32 @comdat() nounwind noinline uwtable "function-instrument"="xray-always" comdat($bar) {
; CHECK: .section .text.comdat,"axG",@progbits,bar,comdat
  ret i32 1
; CHECK: .section xray_instr_map,"aG",@progbits,bar,comdat
}
