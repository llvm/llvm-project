; RUN: llvm-as %s -o %t.o
; RUN: %gold -shared -m elf_x86_64 -o %t2 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN: %t.o --plugin-opt=save-temps
; RUN: llvm-nm %t2 | FileCheck %s --check-prefix=OUTPUT

; OUTPUT: w patatino

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".weak patatino"

declare void @patatino()

define void @_start() {
  call void @patatino()
  ret void
}
