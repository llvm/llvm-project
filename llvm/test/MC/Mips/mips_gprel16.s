// This addresses bug 14456. We were not writing
// out the addend to the gprel16 relocation. The
// addend is stored in the instruction immediate
// field.

// RUN: llvm-mc -mcpu=mips32r2 -triple=mipsel-pc-linux -filetype=obj %s -o - \
// RUN: | llvm-objdump -disassemble - | FileCheck %s
// RUN: llvm-mc -mcpu=mips32r2 -triple=mips-pc-linux -filetype=obj %s -o - \
// RUN: | llvm-objdump -disassemble - | FileCheck %s

	.text
	.abicalls
	.option	pic0
	.section	.mdebug.abi32,"",@progbits
	.file	"/home/espindola/llvm/llvm/test/MC/Mips/mips_gprel16.ll"
	.text
	.globl	testvar1
	.align	2
	.type	testvar1,@function
	.set	nomips16
	.ent	testvar1
testvar1:                               # @testvar1
	.frame	$sp,0,$ra
	.mask 	0x00000000,0
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
	.set	noat
# BB#0:                                 # %entry
// CHECK: lw ${{[0-9]+}}, 0($gp)
	lw	$1, %gp_rel(var1)($gp)
	jr	$ra
	sltu	$2, $zero, $1
	.set	at
	.set	macro
	.set	reorder
	.end	testvar1
$tmp0:
	.size	testvar1, ($tmp0)-testvar1

	.globl	testvar2
	.align	2
	.type	testvar2,@function
	.set	nomips16
	.ent	testvar2
testvar2:                               # @testvar2
	.frame	$sp,0,$ra
	.mask 	0x00000000,0
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
	.set	noat
# BB#0:                                 # %entry
// CHECK: lw ${{[0-9]+}}, 4($gp)
	lw	$1, %gp_rel(var2)($gp)
	jr	$ra
	sltu	$2, $zero, $1
	.set	at
	.set	macro
	.set	reorder
	.end	testvar2
$tmp1:
	.size	testvar2, ($tmp1)-testvar2

	.type	var1,@object            # @var1
	.local	var1
	.comm	var1,4,4
	.type	var2,@object            # @var2
	.local	var2
	.comm	var2,4,4

