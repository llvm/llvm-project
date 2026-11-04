@@ test st_value bit 0 of thumb function
@ RUN: llvm-mc %s -triple=armv4t-freebsd-eabi -filetype=obj -o - | \
@ RUN: llvm-readobj -r  - | FileCheck %s


	.syntax unified
        .text
        .align  2
        .type   f,%function
        .code   16
        .thumb_func
f:
        push    {r7, lr}
        mov     r7, sp
        pop     {r7, pc}

	.section	.data.rel.local,"aw",%progbits
ptr:
	.long	f



@@ make sure the relocation is with f. That is one way to make sure it includes
@@ the thumb bit.
@CHECK:        Section ({{.*}}) .rel.data.rel.local {
@CHECK-NEXT:     0x0 R_ARM_ABS32 f
@CHECK-NEXT:   }
@CHECK-NEXT: ]
