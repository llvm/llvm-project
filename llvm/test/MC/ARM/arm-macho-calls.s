@ RUN: llvm-mc -triple armv7-apple-ios -filetype=obj -o %t %s
@ RUN: llvm-objdump --no-print-imm-hex -d -r %t | FileCheck %s

@ CHECK: <_func>:
@ CHECK:    bl 0x8 <_func+0x8> @ imm = #0
@ CHECK:  ARM_RELOC_BR24 __text
@ CHECK:    bl 0x0 <_func> @ imm = #-12
@ CHECK:  ARM_RELOC_BR24 _elsewhere
    .global _func
_func:
    bl Llocal_symbol
    bl _elsewhere
Llocal_symbol:
    bx lr

    .global _elsewhere
_elsewhere:
    bx lr
