; RUN: llc -mtriple thumbv7-windows-coff -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-COFF
; RUN: llc -mtriple thumbv7-elf -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-OTHER
; RUN: llc -mtriple thumbv7-macho -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-OTHER

declare i8* @f()
declare extern_weak i8* @g(i8*)

define void @test() {
  %call = tail call i8* @f()
  %call1 = tail call i8* @g(i8* %call)
  ret void
}

; CHECK-COFF: movw r0, :lower16:.refptr.g
; CHECK-COFF: movt r0, :upper16:.refptr.g
; CHECK-COFF: ldr r4, [r0]
; CHECK-COFF: mov r1, r4
; CHECK-COFF: bx r1

; CHECK-OTHER: bl {{_?}}g

