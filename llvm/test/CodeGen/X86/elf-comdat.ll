; RUN: llc -mtriple x86_64-pc-linux-gnu < %s | FileCheck %s

$f = comdat any
@v = global i32 0, comdat($f)
define void @f() comdat($f) {
  ret void
}
; CHECK: .section        .text.f,"axG",@progbits,f,comdat
; CHECK: .globl  f
; CHECK: .section        .bss.v,"awG",@nobits,f,comdat
; CHECK: .globl  v
