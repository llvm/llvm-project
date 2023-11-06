; RUN: llvm-as -o %t.bc %s
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext -plugin-opt=emit-asm \
; RUN:    -plugin-opt=asm-verbose \
; RUN:    -m elf_x86_64 -r -o %t.s %t.bc
; RUN: FileCheck --input-file=%t.s %s

; Check if comments are emitted into assembly.
; CHECK: -- End function

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo() {
  ret i32 10
}
