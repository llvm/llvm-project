; RUN: llvm-as %s -o %t1.o
; RUN: llvm-as %p/Inputs/common.ll -o %t2.o
; RUN: llvm-as %p/Inputs/common2.ll -o %t2b.o
; RUN: llvm-as %p/Inputs/common3.ll -o %t2c.o

@a = common global i16 0, align 8

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t1.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s --check-prefix=A

; Shared library case, we merge @a as common and keep it for the symbol table.
; A: @a = common global i32 0, align 8

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t1.o %t2b.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s --check-prefix=B

; (i16 align 8) + (i8 align 16) = i16 align 16
; B: @a = common global i16 0, align 16

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t1.o %t2c.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s --check-prefix=C

; (i16 align 8) + (i8 align 1) = i16 align 8.
; C: @a = common global i16 0, align 8

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    %t1.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck --check-prefix=EXEC %s

; All IR case, we internalize a after merging.
; EXEC: @a = internal global i32 0, align 8

; RUN: llc %p/Inputs/common.ll -o %t2native.o -filetype=obj
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    %t1.o %t2native.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck --check-prefix=MIXED %s

; Mixed ELF and IR. We keep ours as common so the linker will finish the merge.
; MIXED: @a = common global i16 0, align 8
