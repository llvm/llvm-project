; RUN: rm -rf %t && mkdir -p %t
; RUN: llvm-as -o %t/bcsection.bc %s

; RUN: llvm-mc -I=%t -filetype=obj -triple=x86_64-unknown-unknown -o %t/bcsection.bco %p/Inputs/bcsection.s
; RUN: llc -filetype=obj -mtriple=x86_64-unknown-unknown -o %t/bcsection-lib.o %p/Inputs/bcsection-lib.ll

; RUN: %gold -shared --no-undefined -o %t/bcsection.so -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext %t/bcsection.bco %t/bcsection-lib.o

; This test checks that the gold plugin does not attempt to use the bitcode
; in the .llvmbc section for LTO.  bcsection-lib.o calls a function that is
; present the symbol table of bcsection.bco, but not included in the embedded
; bitcode.  If the linker were to use the bitcode, then the symbols in the
; symbol table of bcsection.bco will be ignored and the link will fail.
;
; bcsection.bco:
;  .text:
;    elf_func
;  .llvmbc:
;    bitcode_func
;
; bcsection-lib.o:
;   calls elf_func()

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; CHECK: main
define i32 @bitcode_func() {
  ret i32 0
}
