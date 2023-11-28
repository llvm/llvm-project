# RUN: llvm-mc -triple=armv7-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t.o %s
# RUN: llvm-objdump -r %t.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-objdump --disassemble %t.o | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 -show-entry-es -check %s %t.o


	.text
	.syntax unified

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_CALL call_target_arm
# CHECK-INSTR: 	00000000 <call_site>:
# CHECK-INSTR: 	       0: ebfffffe     bl
# CHECK-INSTR: 	       4: ebfffffe     bl
# CHECK-INSTR: 	0000000c <call_target_arm>
# CHECK-INSTR: 	00000010 <call_target_thumb>
# ARM branch offset is 8, because it accounts for an additional prefetch
# instruction that increments PC even though it is implicit
# jitlink-check: decode_operand(call_site + 0, 0) = call_target_arm   - (call_site +  8)
# jitlink-check: decode_operand(call_site + 4, 0) = call_target_thumb - (call_site + 12)
	.globl	call_site
	.type	call_site,%function
	.p2align	2
call_site:
	bl	call_target_arm
	bl	call_target_thumb
	bx	lr
	.size	call_site, .-call_site

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
# CHECK-INSTR: 	00000014 <jump24_site>:
# CHECK-INSTR: 	      14: eafffffe     b
# CHECK-INSTR: 	00000018 <jump24_target>
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


# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_MOVW_ABS_NC data_symbol
# CHECK-INSTR: 	0000001c <movw>:
# CHECK-INSTR: 	      1c: e3000000     movw      r0, #0x0
# jitlink-check: decode_operand(movw, 1) = (data_symbol&0x0000ffff)
	.globl	movw
	.type	movw,%function
	.p2align	2
movw:
	movw r0, :lower16:data_symbol
	.size	movw,	.-movw

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_MOVT_ABS data_symbol
# CHECK-INSTR: 	00000020 <movt>:
# CHECK-INSTR: 	      20: e3400000     movt      r0, #0x0
# We decode the operand with index 2, because movt generates one leading implicit
# predicate operand that we have to skip in order to decode the data_symbol operand
# jitlink-check: decode_operand(movt, 2) = (data_symbol&0xffff0000>>16)
	.globl	movt
	.type	movt,%function
	.p2align	2
movt:
	movt r0, :upper16:data_symbol
	.size	movt,	.-movt

	.data
	.global data_symbol
data_symbol:
	.long 1073741822

	.text

# Empty main function for jitlink to be happy
	.globl	main
	.type	main,%function
	.p2align	2
main:
	bx	lr
	.size	main,	.-main
