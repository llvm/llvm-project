; REQUIRES: ld_emu_elf32ppc

; RUN: llvm-as %s -o %t.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext -m elf32ppc \
; RUN:    -plugin-opt=mtriple=powerpc-linux-gnu \
; RUN:    -plugin-opt=obj-path=%t3.o \
; RUN:    -shared %t.o -o %t2
; RUN: llvm-readobj --file-headers %t2 | FileCheck  --check-prefix=DSO %s
; RUN: llvm-readobj --file-headers %t3.o | FileCheck --check-prefix=REL %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; REL:       Type: Relocatable
; REL-NEXT:  Machine: EM_PPC

; DSO:       Type: SharedObject
; DSO-NEXT:  Machine: EM_PPC
