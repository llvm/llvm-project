@ RUN: llvm-mc -triple armv7-apple-ios -filetype=obj -o %t %s
@ RUN: llvm-objdump -d -r %t | FileCheck %s

@ CHECK: _func:
@ CHECK:    bl #0 <_func+0x8>
@ CHECK:  ARM_RELOC_BR24 __text
@ CHECK:    bl #-12 <_func>
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
