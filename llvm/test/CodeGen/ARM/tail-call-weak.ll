; RUN: llc -mtriple thumbv7-windows-coff -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-COFF
; RUN: llc -mtriple thumbv7-elf -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-OTHER
; RUN: llc -mtriple thumbv7-macho -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-OTHER

declare i8* @f()
declare extern_weak i8* @g(i8*)

; weak symbol resolution occurs statically in PE/COFF, ensure that we permit
; tail calls on weak externals when targeting a COFF environment.
define void @test() {
  %call = tail call i8* @f()
  %call1 = tail call i8* @g(i8* %call)
  ret void
}

; CHECK-COFF: b g
; CHECK-OTHER: bl {{_?}}g

