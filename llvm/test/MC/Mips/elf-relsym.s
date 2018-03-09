// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux %s -o - | llvm-readobj -t | FileCheck %s

// Check that the appropriate symbols were created.

// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: $.str
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: $.str1
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: $CPI0_0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: $CPI0_1
// CHECK:   }
// CHECK: ]

	.text
	.abicalls
	.section	.mdebug.abi32,"",@progbits
	.file	"/home/espindola/llvm/llvm/test/MC/Mips/elf-relsym.ll"
	.section	.rodata.cst8,"aM",@progbits,8
	.align	3
$CPI0_0:
	.8byte	4612811918334230528     # double 2.5
$CPI0_1:
	.8byte	4616752568008179712     # double 4.5
	.text
	.globl	foo1
	.align	2
	.type	foo1,@function
	.set	nomips16
	.ent	foo1
foo1:                                   # @foo1
	.frame	$sp,0,$ra
	.mask 	0x00000000,0
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
	.set	noat
# %bb.0:                                # %entry
	lui	$2, %hi(_gp_disp)
	addiu	$2, $2, %lo(_gp_disp)
	addu	$1, $2, $25
	lw	$2, %got($.str)($1)
	addiu	$2, $2, %lo($.str)
	lw	$3, %got(gc1)($1)
	sw	$2, 0($3)
	lw	$2, %got($.str1)($1)
	addiu	$2, $2, %lo($.str1)
	lw	$3, %got(gc2)($1)
	sw	$2, 0($3)
	lw	$2, %got($CPI0_0)($1)
	ldc1	$f0, %lo($CPI0_0)($2)
	lw	$2, %got(gd1)($1)
	ldc1	$f2, 0($2)
	lw	$3, %got($CPI0_1)($1)
	ldc1	$f4, %lo($CPI0_1)($3)
	lw	$1, %got(gd2)($1)
	add.d	$f0, $f2, $f0
	sdc1	$f0, 0($2)
	ldc1	$f0, 0($1)
	add.d	$f0, $f0, $f4
	jr	$ra
	sdc1	$f0, 0($1)
	.set	at
	.set	macro
	.set	reorder
	.end	foo1
$tmp0:
	.size	foo1, ($tmp0)-foo1

	.type	$.str,@object           # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
$.str:
	.asciz	"abcde"
	.size	$.str, 6

	.type	$.str1,@object          # @.str1
$.str1:
	.asciz	"fghi"
	.size	$.str1, 5


	.text
