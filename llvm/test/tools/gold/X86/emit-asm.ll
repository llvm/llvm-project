; RUN: llvm-as %s -o %t.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 --plugin-opt=emit-asm \
; RUN:    -shared %t.o -o %t2.s
; RUN: FileCheck --input-file %t2.s %s

; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 --plugin-opt=emit-asm --plugin-opt=lto-partitions=2\
; RUN:    -shared %t.o -o %t2.s
; RUN: cat %t2.s %t2.s1 > %t3.s
; RUN: FileCheck --input-file %t3.s %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-DAG: f1:
define void @f1() {
  ret void
}

; CHECK-DAG: f2:
define void @f2() {
  ret void
}
