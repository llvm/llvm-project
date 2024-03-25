# Test v7 Arm features
#
# RUN: llvm-mc -triple=armv7-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t_armv7.o %s
# RUN: llvm-objdump -r %t_armv7.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-objdump --disassemble %t_armv7.o | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 -abs data_symbol=0x00001234 -check %s %t_armv7.o
#
# RUN: llvm-mc -triple=armv9-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t_armv9.o %s
# RUN: llvm-objdump -r %t_armv9.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-objdump --disassemble %t_armv9.o | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 -abs data_symbol=0x00001234 -check %s %t_armv9.o


	.text
	.syntax unified

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_MOVW_ABS_NC data_symbol
# CHECK-INSTR: <movw>:
# CHECK-INSTR: e3000000 movw r0, #0x0
# jitlink-check: decode_operand(movw, 1) = data_symbol[15:0]
	.globl	movw
	.type	movw,%function
	.p2align	2
movw:
	movw r0, :lower16:data_symbol
	.size	movw,	.-movw

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_MOVT_ABS data_symbol
# CHECK-INSTR: <movt>:
# CHECK-INSTR: e3400000 movt r0, #0x0
# We decode the operand with index 2, because movt generates one leading implicit
# predicate operand that we have to skip in order to decode the data_symbol operand
# jitlink-check: decode_operand(movt, 2) = data_symbol[31:16]
	.globl	movt
	.type	movt,%function
	.p2align	2
movt:
	movt r0, :upper16:data_symbol
	.size	movt,	.-movt

# Empty main function for jitlink to be happy
	.globl	main
	.type	main,%function
	.p2align	2
main:
	bx	lr
	.size	main,	.-main
