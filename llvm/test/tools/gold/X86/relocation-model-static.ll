; RUN: llvm-as %s -o %t.o

;; --noinhibit-exec allows undefined foo.
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --export-dynamic --noinhibit-exec \
; RUN:    --plugin-opt=save-temps %t.o -o %t-out
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=STATIC

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -r \
; RUN:    --plugin-opt=save-temps %t.o -o %t-out
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=STATIC

; STATIC: R_X86_64_PC32 foo

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = external dso_local global i32
define i32 @main() {
  %t = load i32, i32* @foo
  ret i32 %t
}
