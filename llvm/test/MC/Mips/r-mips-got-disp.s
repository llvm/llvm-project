// RUN: llvm-mc -triple=mips64el-pc-linux -filetype=obj -mcpu=mips64r2 < %s -o - | llvm-readobj -r | FileCheck %s

// Check that the R_MIPS_GOT_DISP relocations were created.

//       CHECK: Relocations [
// CHECK:     0x{{[0-9,A-F]+}} R_MIPS_GOT_DISP

	.text
	.abicalls
	.section	.mdebug.abi64,"",@progbits
	.file	"<stdin>"
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
	ld	$1, %got_disp(shl)($gp)
	ld	$5, 0($1)
	ld	$1, %got_page($.str)($gp)
	ld	$25, %call16(printf)($gp)
	jalr	$25
	daddiu	$4, $1, %got_ofst($.str)
	addiu	$2, $zero, 0
	ld	$gp, 0($sp)             # 8-byte Folded Reload
	ld	$ra, 8($sp)             # 8-byte Folded Reload
	jr	$ra
	daddiu	$sp, $sp, 16
	.set	at
	.set	macro
	.set	reorder
	.end	main
$tmp0:
	.size	main, ($tmp0)-main

	.type	shl,@object             # @shl
	.data
	.globl	shl
	.align	3
shl:
	.8byte	1                       # 0x1
	.size	shl, 8

	.type	$.str,@object           # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
$.str:
	.asciz	"0x%llx\n"
	.size	$.str, 8


	.text
