@@ test st_value bit 0 of thumb function
@ RUN: llvm-mc %s -triple=arm-freebsd-eabi -filetype=obj -o - | \
@ RUN: elf-dump  | FileCheck %s


	.syntax unified
        .text
        .globl  f
        .align  2
        .type   f,%function
        .code   16
        .thumb_func
f:
        push    {r7, lr}
        mov     r7, sp
        bl      g
        pop     {r7, pc}

@@ make sure an R_ARM_THM_CALL relocation is generated for the call to g
@CHECK:        ('_relocations', [
@CHECK:         (('r_offset', 0x00000004)
@CHECK-NEXT:     ('r_sym', 0x{{[0-9a-fA-F]+}})
@CHECK-NEXT:     ('r_type', 0x0a)
