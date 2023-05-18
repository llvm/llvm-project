; RUN: llvm-as %s -o %t.o

;; Non-PIC source.

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --shared \
; RUN:    --plugin-opt=save-temps %t.o -o %t-out
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=PIC

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --export-dynamic --noinhibit-exec -pie \
; RUN:    --plugin-opt=save-temps %t.o -o %t-out
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=PIC

;; PIC source.

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --shared \
; RUN:    --plugin-opt=save-temps %t.o -o %t-out
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=PIC

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --export-dynamic --noinhibit-exec -pie \
; RUN:    --plugin-opt=save-temps %t.o -o %t-out
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=PIC

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -r \
; RUN:    --plugin-opt=save-temps %t.o -o %t-out
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=PIC


; PIC: R_X86_64_GOTPCREL foo

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = external global i32
define i32 @main() {
  %t = load i32, ptr @foo
  ret i32 %t
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"PIC Level", i32 2}
