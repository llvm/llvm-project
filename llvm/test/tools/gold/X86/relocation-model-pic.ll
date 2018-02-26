; RUN: cat %s >%t.pic.ll
; RUN: echo '!llvm.module.flags = !{!0}' >>%t.pic.ll
; RUN: echo '!0 = !{i32 1, !"PIC Level", i32 2}' >>%t.pic.ll

; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %t.pic.ll -o %t.pic.o

;; Non-PIC source.

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --shared \
; RUN:    --plugin-opt=save-temps %t.o -o %t-out
; RUN: llvm-readobj -r %t-out.o | FileCheck %s --check-prefix=PIC

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --export-dynamic --noinhibit-exec -pie \
; RUN:    --plugin-opt=save-temps %t.o -o %t-out
; RUN: llvm-readobj -r %t-out.o | FileCheck %s --check-prefix=PIC

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --export-dynamic --noinhibit-exec \
; RUN:    --plugin-opt=save-temps %t.o -o %t-out
; RUN: llvm-readobj -r %t-out.o | FileCheck %s --check-prefix=STATIC

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -r \
; RUN:    --plugin-opt=save-temps %t.o -o %t-out
; RUN: llvm-readobj -r %t-out.o | FileCheck %s --check-prefix=STATIC

;; PIC source.

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --shared \
; RUN:    --plugin-opt=save-temps %t.pic.o -o %t-out
; RUN: llvm-readobj -r %t-out.o | FileCheck %s --check-prefix=PIC

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --export-dynamic --noinhibit-exec -pie \
; RUN:    --plugin-opt=save-temps %t.pic.o -o %t-out
; RUN: llvm-readobj -r %t-out.o | FileCheck %s --check-prefix=PIC

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --export-dynamic --noinhibit-exec \
; RUN:    --plugin-opt=save-temps %t.pic.o -o %t-out
; RUN: llvm-readobj -r %t-out.o | FileCheck %s --check-prefix=STATIC

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -r \
; RUN:    --plugin-opt=save-temps %t.pic.o -o %t-out
; RUN: llvm-readobj -r %t-out.o | FileCheck %s --check-prefix=PIC


; PIC: R_X86_64_GOTPCREL foo
; STATIC: R_X86_64_PC32 foo

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = external global i32
define i32 @main() {
  %t = load i32, i32* @foo
  ret i32 %t
}
