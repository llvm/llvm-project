@ RUN: llvm-mc -filetype=obj --triple=thumbv6m-none-eabi %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC
@ RUN: llvm-objdump -d --triple=thumbv6m-none-eabi %t | FileCheck %s --check-prefix=ADDEND

    .section .text._func1, "ax"

    .balign 4
    .global _func1
    .type _func1, %function
_func1:
    adr r0, _func2
@ RELOC: R_ARM_THM_PC8
    bx lr

// Checking the encoding only, as the disassembly is not quite correct here.
//00000000 <_func1>:
//       0: a0ff         	adr	r0, #1020 <_func1+0x103>

// Thumb16 encoding supports only adding of the encoded immediate (not
// subtracting, see [Arm ARM]), therefore sign change is required if the pcrel
// offset is negative. This makes the calculation of the addend for
// R_ARM_THM_PC8 more complex, for details see [ELF for the Arm 32-bit
// architecture].

@ ADDEND: a0ff adr

