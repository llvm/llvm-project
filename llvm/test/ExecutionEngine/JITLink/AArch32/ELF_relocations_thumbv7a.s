# Test v7 Thumb features for mixed Arm/Thumb targets
#
# RUN: llvm-mc -triple=thumbv7-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t_thumbv7.o %s
# RUN: llvm-objdump -r %t_thumbv7.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-objdump --disassemble %t_thumbv7.o | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 -abs external_func=0x76bbe880 \
# RUN:              -check %s %t_thumbv7.o


	.text
	.syntax unified

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_THM_CALL call_target_arm
# CHECK-INSTR: <call_site>:
# CHECK-INSTR: f7ff fffe     bl
# We decode the operand with index 2, because bl generates two leading implicit
# predicate operands that we have to skip in order to decode the call_target operand
# jitlink-check: decode_operand(call_site, 2) = call_target_arm - next_pc(call_site)
	.globl	call_site
	.type	call_site,%function
	.p2align	1
	.code	16
	.thumb_func
call_site:
	bl	call_target_arm
	.size	call_site, .-call_site

	.globl	call_target_arm
	.type	call_target_arm,%function
	.p2align	2
	.code	32
call_target_arm:
	bx	lr
	.size	call_target_arm, .-call_target_arm

# Empty main function for jitlink to be happy
	.globl	main
	.type	main,%function
	.p2align	1
	.code	16
	.thumb_func
main:
	bx	lr
	.size	main,	.-main
