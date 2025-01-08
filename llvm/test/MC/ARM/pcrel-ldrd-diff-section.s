@ RUN: not llvm-mc -filetype=obj --defsym=ERR=1 -o /dev/null %s 2>&1 -triple=thumbv7   | FileCheck %s --check-prefix=ERR
@ RUN: not llvm-mc -filetype=obj --defsym=ERR=1 -o /dev/null %s 2>&1 -triple=thumbebv7 | FileCheck %s --check-prefix=ERR
@ RUN: llvm-mc -filetype=obj -triple=armv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=ARM
@ RUN: llvm-objdump -d --triple=armv7 %t | FileCheck %s --check-prefix=ARM_ADDEND
@ RUN: llvm-mc -filetype=obj -triple=armebv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=ARM
@ RUN: llvm-objdump -d --triple=armebv7 %t | FileCheck %s --check-prefix=ARM_ADDEND

    .section .text.bar, "ax"
    .balign 4
    .global bar
    .type bar, %function

bar:
    ldrd r0, r1, foo1    @ arm_pcrel_10_unscaled
    ldrd r0, r1, foo2-8  @ arm_pcrel_10_unscaled
.ifdef ERR
  @ ERR:[[#@LINE-3]]:5: error: unsupported relocation type
  @ ERR:[[#@LINE-3]]:5: error: unsupported relocation type
.endif
    bx lr

    .section .data.foo, "a", %progbits
    .balign 4
    .global foo1
    .global foo2
foo1:
    .word 0x11223344, 0x55667788
foo2:
    .word 0x99aabbcc, 0xddeeff00

@ ARM: R_ARM_LDRS_PC_G0

@ ARM_ADDEND: ldrd r0, r1, [pc, #-8]
@ ARM_ADDEND: ldrd r0, r1, [pc, #-16]
