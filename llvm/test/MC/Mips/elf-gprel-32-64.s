// RUN: llvm-mc -filetype=obj -triple=mips64el-pc-linux -mcpu=mips64 %s -o - \
// RUN: | llvm-readobj -r \
// RUN: | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple=mips64-pc-linux -mcpu=mips64 %s -o - \
// RUN: | llvm-readobj -r \
// RUN: | FileCheck %s

// Check that the appropriate relocations were created.

// R_MIPS_GPREL32/R_MIPS_64/R_MIPS_NONE
// CHECK:      Relocations [
// CHECK:        Section ({{[a-z0-9]+}}) .rela.rodata {
// CHECK-NEXT:     0x{{[0-9,A-F]+}} R_MIPS_GPREL32/R_MIPS_64/R_MIPS_NONE
// CHECK-NEXT:     0x{{[0-9,A-F]+}} R_MIPS_GPREL32/R_MIPS_64/R_MIPS_NONE
// CHECK-NEXT:     0x{{[0-9,A-F]+}} R_MIPS_GPREL32/R_MIPS_64/R_MIPS_NONE
// CHECK-NEXT:     0x{{[0-9,A-F]+}} R_MIPS_GPREL32/R_MIPS_64/R_MIPS_NONE
// CHECK-NEXT:   }
// CHECK-NEXT: ]

	.text
	.abicalls
	.section	.mdebug.abi64,"",@progbits
	.file	"/home/espindola/llvm/llvm/test/MC/Mips/elf-gprel-32-64.ll"
	.text
	.globl	test
	.align	3
	.type	test,@function
	.set	nomips16
	.ent	test
test:                                   # @test
	.frame	$sp,0,$ra
	.mask 	0x00000000,0
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
	.set	noat
# %bb.0:                                # %entry
	lui	$1, %hi(%neg(%gp_rel(test)))
	daddu	$2, $1, $25
	sltiu	$1, $4, 4
	dsll	$3, $4, 32
	bnez	$1, $BB0_2
	nop
$BB0_1:                                 # %sw.default
	b	$BB0_3
	addiu	$2, $zero, -1
$BB0_2:                                 # %entry
	daddiu	$1, $2, %lo(%neg(%gp_rel(test)))
	dsrl	$3, $3, 32
	daddiu	$4, $zero, 8
	dmult	$3, $4
	mflo	$3
	ld	$4, %got_page($JTI0_0)($1)
	daddu	$3, $3, $4
	ld	$3, %got_ofst($JTI0_0)($3)
	daddu	$1, $3, $1
	jr	$1
	addiu	$2, $zero, 1
$BB0_3:                                 # %return
	jr	$ra
	nop
$BB0_4:                                 # %sw.bb2
	jr	$ra
	addiu	$2, $zero, 3
$BB0_5:                                 # %sw.bb5
	jr	$ra
	addiu	$2, $zero, 2
$BB0_6:                                 # %sw.bb8
	jr	$ra
	addiu	$2, $zero, 7
	.set	at
	.set	macro
	.set	reorder
	.end	test
$tmp0:
	.size	test, ($tmp0)-test
	.section	.rodata,"a",@progbits
	.align	3
$JTI0_0:
	.gpdword	($BB0_3)
	.gpdword	($BB0_4)
	.gpdword	($BB0_5)
	.gpdword	($BB0_6)


	.text
