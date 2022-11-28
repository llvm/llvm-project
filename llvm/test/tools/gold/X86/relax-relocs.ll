; RUN: llvm-as %s -o %t.o
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=save-temps \
; RUN:    -shared %t.o -o %t.so
; RUN: llvm-readobj -r %t.so.lto.o | FileCheck %s

; Test that we produce R_X86_64_GOTPCREL instead of R_X86_64_GOTPCRELX
; CHECK: R_X86_64_GOTPCREL foo

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = external global i32
define i32 @bar() {
  %t = load i32, ptr @foo
  ret i32 %t
}
