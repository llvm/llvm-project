// This test case will cause an internal EK_GPRel64BlockAddress to be
// produced. This was not handled for direct object and an assertion
// to occur. This is a variation on test case test/CodeGen/Mips/do_switch.ll

// RUN: llvm-mc < %s -filetype=obj -triple=mips-pc-linux

	.text
	.abicalls
	.section	.mdebug.abi32,"",@progbits
	.file	"/home/espindola/llvm/llvm/test/MC/Mips/do_switch.ll"
	.text
	.globl	main
	.align	2
	.type	main,@function
	.set	nomips16
	.ent	main
main:                                   # @main
	.frame	$sp,8,$ra
	.mask 	0x00000000,0
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
	.set	noat
# BB#0:                                 # %entry
	lui	$2, %hi(_gp_disp)
	addiu	$2, $2, %lo(_gp_disp)
	addiu	$sp, $sp, -8
	addiu	$1, $zero, 2
	sw	$1, 4($sp)
	lw	$3, 4($sp)
	sltiu	$1, $3, 4
	bnez	$1, $BB0_2
	addu	$2, $2, $25
$BB0_1:                                 # %bb4
	addiu	$2, $zero, 4
	jr	$ra
	addiu	$sp, $sp, 8
$BB0_2:                                 # %entry
	sll	$1, $3, 2
	lw	$3, %got($JTI0_0)($2)
	addu	$1, $1, $3
	lw	$1, %lo($JTI0_0)($1)
	addu	$1, $1, $2
	jr	$1
	nop
$BB0_3:                                 # %bb5
	addiu	$2, $zero, 1
	jr	$ra
	addiu	$sp, $sp, 8
$BB0_4:                                 # %bb1
	addiu	$2, $zero, 2
	jr	$ra
	addiu	$sp, $sp, 8
$BB0_5:                                 # %bb2
	addiu	$2, $zero, 0
	jr	$ra
	addiu	$sp, $sp, 8
$BB0_6:                                 # %bb3
	addiu	$2, $zero, 3
	jr	$ra
	addiu	$sp, $sp, 8
	.set	at
	.set	macro
	.set	reorder
	.end	main
$tmp0:
	.size	main, ($tmp0)-main
	.section	.rodata,"a",@progbits
	.align	2
$JTI0_0:
	.gpword	($BB0_3)
	.gpword	($BB0_4)
	.gpword	($BB0_5)
	.gpword	($BB0_6)


	.text
