// This test case will cause an internal EK_GPRel64BlockAddress to be
// produced. This was not handled for direct object and an assertion
// to occur. This is a variation on test case test/CodeGen/Mips/do_switch.ll

// RUN: llvm-mc < %s -filetype=obj -triple=mips64-pc-linux -mcpu=mips64 -target-abi=n64

	.text
	.abicalls
	.section	.mdebug.abi64,"",@progbits
	.file	"/home/espindola/llvm/llvm/test/MC/Mips/do_switch.ll"
	.text
	.globl	main
	.align	3
	.type	main,@function
	.set	nomips16
	.ent	main
main:                                   # @main
	.frame	$sp,16,$ra
	.mask 	0x00000000,0
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
	.set	noat
# %bb.0:                                # %entry
	daddiu	$sp, $sp, -16
	lui	$1, %hi(%neg(%gp_rel(main)))
	daddu	$2, $1, $25
	addiu	$1, $zero, 2
	sw	$1, 12($sp)
	lw	$1, 12($sp)
	sltiu	$4, $1, 4
	dsll	$3, $1, 32
	bnez	$4, $BB0_2
	nop
$BB0_1:                                 # %bb4
	addiu	$2, $zero, 4
	jr	$ra
	daddiu	$sp, $sp, 16
$BB0_2:                                 # %entry
	daddiu	$1, $2, %lo(%neg(%gp_rel(main)))
	dsrl	$2, $3, 32
	daddiu	$3, $zero, 8
	dmult	$2, $3
	mflo	$2
	ld	$3, %got_page($JTI0_0)($1)
	daddu	$2, $2, $3
	ld	$2, %got_ofst($JTI0_0)($2)
	daddu	$1, $2, $1
	jr	$1
	nop
$BB0_3:                                 # %bb5
	addiu	$2, $zero, 1
	jr	$ra
	daddiu	$sp, $sp, 16
$BB0_4:                                 # %bb1
	addiu	$2, $zero, 2
	jr	$ra
	daddiu	$sp, $sp, 16
$BB0_5:                                 # %bb2
	addiu	$2, $zero, 0
	jr	$ra
	daddiu	$sp, $sp, 16
$BB0_6:                                 # %bb3
	addiu	$2, $zero, 3
	jr	$ra
	daddiu	$sp, $sp, 16
	.set	at
	.set	macro
	.set	reorder
	.end	main
$tmp0:
	.size	main, ($tmp0)-main
	.section	.rodata,"a",@progbits
	.align	3
$JTI0_0:
	.gpdword	($BB0_3)
	.gpdword	($BB0_4)
	.gpdword	($BB0_5)
	.gpdword	($BB0_6)


	.text
