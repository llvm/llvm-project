// RUN: llvm-mc -filetype=obj -triple=mips64el-pc-linux -mcpu=mips64  %s -o - | llvm-readobj -r - | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple=mips64-pc-linux -mcpu=mips64  %s -o - | llvm-readobj -r - | FileCheck %s

// Check for N64 relocation production.
// Check that the appropriate relocations were created.

// CHECK: Relocations [
// CHECK:   0x{{[0-9,A-F]+}} R_MIPS_GPREL16/R_MIPS_SUB/R_MIPS_HI16
// CHECK:   0x{{[0-9,A-F]+}} R_MIPS_GPREL16/R_MIPS_SUB/R_MIPS_LO16
// CHECK:   0x{{[0-9,A-F]+}} R_MIPS_GOT_PAGE/R_MIPS_NONE/R_MIPS_NONE
// CHECK:   0x{{[0-9,A-F]+}} R_MIPS_GOT_OFST/R_MIPS_NONE/R_MIPS_NONE
// CHECK: ]


	.text
	.abicalls
	.section	.mdebug.abi64,"",@progbits
	.file	"/home/espindola/llvm/llvm/test/MC/Mips/elf-N64.ll"
	.text
	.globl	main
	.align	3
	.type	main,@function
	.set	nomips16
	.ent	main
main:                                   # @main
	.frame	$sp,16,$ra
	.mask 	0x00000000,0
	.fmask	0x90000000,-4
	.set	noreorder
	.set	nomacro
	.set	noat
# %bb.0:                                # %entry
	daddiu	$sp, $sp, -16
	sd	$ra, 8($sp)             # 8-byte Folded Spill
	sd	$gp, 0($sp)             # 8-byte Folded Spill
	lui	$1, %hi(%neg(%gp_rel(main)))
	daddu	$1, $1, $25
	daddiu	$gp, $1, %lo(%neg(%gp_rel(main)))
	ld	$1, %got_page($str)($gp)
	daddiu	$4, $1, %got_ofst($str)
	ld	$25, %call16(puts)($gp)
	jalr	$25
	nop
	addiu	$2, $zero, 0
	ld	$gp, 0($sp)             # 8-byte Folded Reload
	ld	$ra, 8($sp)             # 8-byte Folded Reload
	daddiu	$sp, $sp, 16
	jr	$ra
	nop
	.set	at
	.set	macro
	.set	reorder
	.end	main
$tmp0:
	.size	main, ($tmp0)-main

	.type	$str,@object            # @str
	.section	.rodata.str1.4,"aMS",@progbits,1
	.align	2
$str:
	.asciz	"hello world"
	.size	$str, 12


	.text
