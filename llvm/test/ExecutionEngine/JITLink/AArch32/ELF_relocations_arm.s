# Test pre-v7 Arm features
#
# RUN: llvm-mc -triple=armv4t-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t_armv4t.o %s
# RUN: llvm-objdump -r %t_armv4t.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-objdump --disassemble %t_armv4t.o | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 -check %s %t_armv4t.o
#
# RUN: llvm-mc -triple=armv7-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t_armv7.o %s
# RUN: llvm-objdump -r %t_armv7.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-objdump --disassemble %t_armv7.o | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 -check %s %t_armv7.o
#
# RUN: llvm-mc -triple=armv9-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t_armv9.o %s
# RUN: llvm-objdump -r %t_armv9.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-objdump --disassemble %t_armv9.o | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 -check %s %t_armv9.o


	.text
	.syntax unified

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_CALL call_target_arm
# CHECK-INSTR: 	00000000 <call_site_a2a>:
# CHECK-INSTR: 	       0: ebfffffe     bl
# CHECK-INSTR: 	00000008 <call_site_a2t>:
# CHECK-INSTR: 	       8: ebfffffe     bl
# CHECK-INSTR: 	00000010 <call_target_arm>
# CHECK-INSTR: 	00000014 <call_target_thumb>
# jitlink-check: decode_operand(call_site_a2a, 0) = call_target_arm   - next_pc(call_site_a2a)
# jitlink-check: decode_operand(call_site_a2t, 0) = call_target_thumb - next_pc(call_site_a2t)
	.globl	call_site_a2a
	.type	call_site_a2a, %function
	.p2align	2
call_site_a2a:
	bl	call_target_arm
	bx	lr
	.size	call_site_a2a, .-call_site_a2a

	.globl	call_site_a2t
	.type	call_site_a2t,%function
	.p2align	2
call_site_a2t:
	bl	call_target_thumb
	bx	lr
	.size	call_site_a2t, .-call_site_a2t

	.globl	call_target_arm
	.type	call_target_arm,%function
	.p2align	2
call_target_arm:
	bx	lr
	.size	call_target_arm, .-call_target_arm

	.code	16
	.globl	call_target_thumb
	.type	call_target_thumb,%function
	.p2align	1
	.thumb_func
call_target_thumb:
	bx	lr
	.size	call_target_thumb, .-call_target_thumb
	.code 32

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_JUMP24 jump24_target
# CHECK-INSTR: 	00000018 <jump24_site>:
# CHECK-INSTR: 	      18: eafffffe     b
# CHECK-INSTR: 	0000001c <jump24_target>
# jitlink-check: decode_operand(jump24_site, 0) = jump24_target - next_pc(jump24_site)
	.globl	jump24_site
	.type	jump24_site,%function
	.p2align	2
jump24_site:
	b.w	jump24_target
	.size	jump24_site,	.-jump24_site

	.globl	jump24_target
	.type	jump24_target,%function
	.p2align	2
jump24_target:
	bx	lr
	.size	jump24_target,	.-jump24_target

# Empty main function for jitlink to be happy
	.globl	main
	.type	main,%function
	.p2align	2
main:
	bx	lr
	.size	main,	.-main
