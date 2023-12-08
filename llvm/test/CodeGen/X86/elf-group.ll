; Checks that comdat with nodeduplicate kind is lowered to a zero-flag ELF
; section group.
; RUN: llc < %s -mtriple=x86_64-unknown-linux | FileCheck %s

; CHECK: .section .text.f1,"axG",@progbits,f1{{$}}
; CHECK: .section .text.f2,"axG",@progbits,f1{{$}}
; CHECK: .section .bss.g1,"aGw",@nobits,f1{{$}}

$f1 = comdat nodeduplicate

define void @f1() comdat {
  unreachable
}

define hidden void @f2() comdat($f1) {
  unreachable
}

@g1 = global i32 0, comdat($f1)
