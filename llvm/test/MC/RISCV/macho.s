; RUN: llvm-mc -triple riscv32-apple-macho %s -o - | FileCheck %s
; RUN: llvm-mc -triple riscv32-apple-macho -filetype=obj %s -o %t.o
; RUN: llvm-objdump -d %t.o | FileCheck %s --check-prefix=CHECK-DIS
; RUN: llvm-nm %t.o | FileCheck %s --check-prefix=CHECK-SYMS

        nop
        .half 42
        .word 42
Lfoo:
lfoo:
foo:

; CHECK: nop
; CHECK: .half 42
; CHECK: .word 42

; CHECK-DIS: file format mach-o 32-bit risc-v
; CHECK-DIS: Disassembly of section __TEXT,__text:
; CHECK-DIS: nop
; CHECK-DIS: 002a 
; CHECK-DIS: 002a
; CHECK-DIS: 0000

; CHECK-SYMS-NOT: Lfoo
; CHECK-SYMS: foo
; CHECK-SYMS: lfoo
