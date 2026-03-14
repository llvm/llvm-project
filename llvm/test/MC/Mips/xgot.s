// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux  %s -o - | llvm-readobj -r - | FileCheck %s

// Check that the appropriate relocations were created.
// For the xgot case we want to see R_MIPS_[GOT|CALL]_[HI|LO]16.

// CHECK: Relocations [
// CHECK:   0x0 R_MIPS_HI16 _gp_disp
// CHECK:   0x4 R_MIPS_LO16 _gp_disp
// CHECK:   0x14 R_MIPS_GOT_HI16 ext_1
// CHECK:   0x1C R_MIPS_GOT_LO16 ext_1
// CHECK:   0x24 R_MIPS_CALL_HI16 printf
// CHECK:   0x30 R_MIPS_CALL_LO16 printf
// CHECK:   0x2C R_MIPS_GOT16 $.str
// CHECK:   0x38 R_MIPS_LO16 $.str
// CHECK: ]

	.text
	.abicalls
	.section	.mdebug.abi32,"",@progbits
	.file	"/home/espindola/llvm/llvm/test/MC/Mips/xgot.ll"
	.text
	.globl	fill
	.align	2
	.type	fill,@function
	.set	nomips16
	.ent	fill
fill:                                   # @fill
	.frame	$sp,24,$ra
	.mask 	0x80000000,-4
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
	.set	noat
# %bb.0:                                # %entry
	lui	$2, %hi(_gp_disp)
	addiu	$2, $2, %lo(_gp_disp)
	addiu	$sp, $sp, -24
	sw	$ra, 20($sp)            # 4-byte Folded Spill
	addu	$gp, $2, $25
	lui	$1, %got_hi(ext_1)
	addu	$1, $1, $gp
	lw	$1, %got_lo(ext_1)($1)
	lw	$5, 0($1)
	lui	$1, %call_hi(printf)
	addu	$1, $1, $gp
	lw	$2, %got($.str)($gp)
	lw	$25, %call_lo(printf)($1)
	jalr	$25
	addiu	$4, $2, %lo($.str)
	lw	$ra, 20($sp)            # 4-byte Folded Reload
	jr	$ra
	addiu	$sp, $sp, 24
	.set	at
	.set	macro
	.set	reorder
	.end	fill
$tmp0:
	.size	fill, ($tmp0)-fill

	.type	$.str,@object           # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
$.str:
	.asciz	"ext_1=%d, i=%d\n"
	.size	$.str, 16


	.text
