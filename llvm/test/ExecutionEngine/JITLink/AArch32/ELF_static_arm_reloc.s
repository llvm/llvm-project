# RUN: llvm-mc -triple=armv7-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t.o %s
# RUN: llvm-objdump -r %t.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-objdump --disassemble %t.o | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 -show-entry-es -check %s %t.o


	.text
	.syntax unified

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_CALL call_target
# CHECK-INSTR: 	00000000 <call_site>:
# CHECK-INSTR: 	       0: ebfffffe     bl      0x0 <call_site>
# CHECK-INSTR: 	00000004 <call_target>:
# CHECK-INSTR: 	       4: e12fff1e     bx      lr
# ARM branch offset is 8, because it accounts for an additional prefetch
# instruction that increments PC even though it is implicit
# jitlink-check: decode_operand(call_site, 0) = call_target - (call_site + 8)
	.globl	call_site
	.type	call_site,%function
	.p2align	2
call_site:
	bl	call_target
	.size	call_site,	.-call_site

	.globl	call_target
	.type	call_target,%function
	.p2align	2
call_target:
	bx	lr
	.size	call_target,	.-call_target

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_JUMP24 jump24_target
# CHECK-INSTR: 	00000008 <jump24_site>:
# CHECK-INSTR: 	       8: eafffffe     b      0x8 <jump24_site>
# CHECK-INSTR: 	0000000c <jump24_target>:
# CHECK-INSTR: 	       c: e12fff1e     bx      lr
# jitlink-check: decode_operand(jump24_site, 0) = jump24_target - (jump24_site + 8)
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
