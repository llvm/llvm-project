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
        bl      g
        pop     {r7, pc}

	.section	.data.rel.local,"aw",%progbits
ptr:
	.long	f


@@ make sure an R_ARM_THM_CALL relocation is generated for the call to g
@CHECK:      Relocations [
@CHECK-NEXT:   Section {{.*}} .rel.text {
@CHECK-NEXT:     0x4 R_ARM_THM_CALL g
@CHECK-NEXT:   }


@@ make sure the relocation is with f. That is one way to make sure it includes
@@ the thumb bit.
@CHECK-NEXT:   Section ({{.*}}) .rel.data.rel.local {
@CHECK-NEXT:     0x0 R_ARM_ABS32 f
@CHECK-NEXT:   }
@CHECK-NEXT: ]
