# Test v7 Thumb features for Thumb-only targets
#
# RUN: llvm-mc -triple=thumbv7m-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t_thumbv7m.o %s
# RUN: llvm-objdump -r %t_thumbv7m.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-objdump --disassemble %t_thumbv7m.o | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 -abs ext_func=0x76bbe880 -abs ext_data=0x00001234 \
# RUN:              -check %s %t_thumbv7m.o
#
# RUN: llvm-mc -triple=thumbv7-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t_thumbv7.o %s
# RUN: llvm-objdump -r %t_thumbv7.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-objdump --disassemble %t_thumbv7.o | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 -abs ext_func=0x76bbe880 -abs ext_data=0x00001234 \
# RUN:              -check %s %t_thumbv7.o

	.text
	.syntax unified


# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_THM_JUMP24 jump24_target
# CHECK-INSTR: <jump24_site>:
# CHECK-INSTR: f7ff bffe     b.w
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
	.size	jump24_site, .-jump24_site

	.globl	jump24_target
	.type	jump24_target,%function
	.p2align	1
	.code	16
	.thumb_func
jump24_target:
	bx	lr
	.size	jump24_target, .-jump24_target

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_THM_MOVW_ABS_NC ext_data
# CHECK-INSTR: <movw>:
# CHECK-INSTR: f240 0000     movw    r0, #0x0
# jitlink-check: decode_operand(movw, 1) = ext_data[15:0]
	.globl	movw
	.type	movw,%function
	.p2align	1
	.code	16
	.thumb_func
movw:
	movw r0, :lower16:ext_data
	.size	movw,	.-movw

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_THM_MOVT_ABS ext_data
# CHECK-INSTR: <movt>:
# CHECK-INSTR: f2c0 0000     movt    r0, #0x0
# We decode the operand with index 2, because movt generates one leading implicit
# predicate operand that we have to skip in order to decode the ext_data operand
# jitlink-check: decode_operand(movt, 2) = ext_data[31:16]
	.globl	movt
	.type	movt,%function
	.p2align	1
	.code	16
	.thumb_func
movt:
	movt r0, :upper16:ext_data
	.size	movt,	.-movt

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_THM_MOVW_PREL_NC ext_func
# CHECK-INSTR: <movw_prel>:
# CHECK-INSTR: f240 0000     movw    r0, #0x0
# jitlink-check: decode_operand(movw_prel, 1) = (ext_func - movw_prel)[15:0]
  .globl	movw_prel
  .type	movw_prel,%function
  .p2align	1
  .code	16
  .thumb_func
movw_prel:
	movw r0, :lower16:ext_func - .
	.size	movw_prel, .-movw_prel

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_THM_MOVT_PREL ext_func
# CHECK-INSTR: <movt_prel>:
# CHECK-INSTR: f2c0 0000    movt    r0, #0x0
# jitlink-check: decode_operand(movt_prel, 2) = (ext_func - movt_prel)[31:16]
  .globl	movt_prel
  .type	movt_prel,%function
  .p2align	1
  .code	16
  .thumb_func
movt_prel:
	movt r0, :upper16:ext_func - .
	.size	movt_prel, .-movt_prel

# Empty main function for jitlink to be happy
	.globl	main
	.type	main,%function
	.p2align	1
	.code	16
	.thumb_func
main:
	bx	lr
	.size	main,	.-main
