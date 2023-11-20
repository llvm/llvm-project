# RUN: llvm-mc -triple=thumbv7-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t.o %s
# RUN: llvm-objdump -r %t.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-objdump --disassemble %t.o | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 -abs external_func=0x76bbe880 \
# RUN:              -check %s %t.o


	.text
	.syntax unified

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_THM_CALL call_target
# CHECK-INSTR: 	00000000 <call_site>:
# CHECK-INSTR: 	       0: f7ff fffe     bl      0x0 <call_site>
# CHECK-INSTR: 	00000004 <call_target>:
# CHECK-INSTR: 	       4: 4770          bx      lr
# We decode the operand with index 2, because bl generates two leading implicit
# predicate operands that we have to skip in order to decode the call_target operand
# jitlink-check: decode_operand(call_site, 2) = call_target - next_pc(call_site)
	.globl	call_site
	.type	call_site,%function
	.p2align	1
	.code	16
	.thumb_func
call_site:
	bl	call_target
	.size	call_site,	.-call_site

	.globl	call_target
	.type	call_target,%function
	.p2align	1
	.code	16
	.thumb_func
call_target:
	bx	lr
	.size	call_target,	.-call_target

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_THM_JUMP24 jump24_target
# CHECK-INSTR: 	00000006 <jump24_site>:
# CHECK-INSTR: 	       6: f7ff bffe     b.w     0x6 <jump24_site>
# CHECK-INSTR: 	0000000a <jump24_target>:
# CHECK-INSTR: 	       a: 4770          bx      lr
# b.w generates two implicit predicate operands as well, but they are trailing
# operands, so there is no need to adjust the operand index.
# jitlink-check: decode_operand(jump24_site, 0) = jump24_target - next_pc(jump24_site)
	.globl	jump24_site
	.type	jump24_site,%function
	.p2align	1
	.code	16
	.thumb_func
jump24_site:
	b.w	jump24_target
	.size	jump24_site,	.-jump24_site

	.globl	jump24_target
	.type	jump24_target,%function
	.p2align	1
	.code	16
	.thumb_func
jump24_target:
	bx	lr
	.size	jump24_target,	.-jump24_target

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_THM_MOVW_ABS_NC data_symbol
# CHECK-INSTR: 	0000000c <movw>:
# CHECK-INSTR: 	       c: f240 0000     movw    r0, #0x0
# jitlink-check: decode_operand(movw, 1) = (data_symbol&0x0000ffff)
	.globl	movw
	.type	movw,%function
	.p2align	1
	.code	16
	.thumb_func
movw:
	movw r0, :lower16:data_symbol
	.size	movw,	.-movw

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_THM_MOVT_ABS data_symbol
# CHECK-INSTR: 	00000010 <movt>:
# CHECK-INSTR: 	      10: f2c0 0000     movt    r0, #0x0
# We decode the operand with index 2, because movt generates one leading implicit
# predicate operand that we have to skip in order to decode the data_symbol operand
# jitlink-check: decode_operand(movt, 2) = (data_symbol&0xffff0000>>16)
	.globl	movt
	.type	movt,%function
	.p2align	1
	.code	16
	.thumb_func
movt:
	movt r0, :upper16:data_symbol
	.size	movt,	.-movt

	.data
	.global data_symbol
data_symbol:
	.long 1073741822

	.text

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_THM_MOVW_PREL_NC external_func
# CHECK-INSTR: 	00000014 <movw_prel>:
# CHECK-INSTR: 	      14: f240 0000     movw    r0, #0x0
# jitlink-check: decode_operand(movw_prel, 1) = \
# jitlink-check:              ((external_func - movw_prel)&0x0000ffff)
.globl	movw_prel
.type	movw_prel,%function
.p2align	1
.code	16
.thumb_func
movw_prel:
	movw r0, :lower16:external_func - .
	.size	movw_prel,	.-movw_prel

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_THM_MOVT_PREL external_func 
# CHECK-INSTR: 	00000018 <movt_prel>:
# CHECK-INSTR: 	      18: f2c0 0000    movt    r0, #0x0
# jitlink-check: decode_operand(movt_prel, 2) = \
# jitlink-check:               ((external_func - movt_prel)&0xffff0000>>16)
.globl	movt_prel
.type	movt_prel,%function
.p2align	1
.code	16
.thumb_func
movt_prel:
	movt r0, :upper16:external_func - .
	.size	movt_prel,	.-movt_prel

# Empty main function for jitlink to be happy
	.globl	main
	.type	main,%function
	.p2align	1
	.code	16
	.thumb_func
main:
	bx	lr
	.size	main,	.-main
