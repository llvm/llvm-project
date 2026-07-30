// RUN: llvm-mc -triple=armebv8.2a-eabi -filetype=obj < %s | llvm-objdump -s - | FileCheck %s --check-prefix=CHECK-BE
// RUN: llvm-mc -triple=armv8.2a-eabi -filetype=obj < %s | llvm-objdump -s - | FileCheck %s --check-prefix=CHECK-LE

	.text
	.fpu	crypto-neon-fp-armv8
        .arch_extension fp16

.section s_pcrel_9,"ax",%progbits
// CHECK-BE-LABEL: Contents of section s_pcrel_9
// CHECK-LE-LABEL: Contents of section s_pcrel_9
// CHECK-BE: 0000 ed9f0902
// CHECK-LE: 0000 02099fed
	vldr.16	s0, .LCPI0_0
        nop
        bx lr
.LCPI0_0:
	.short	28012

