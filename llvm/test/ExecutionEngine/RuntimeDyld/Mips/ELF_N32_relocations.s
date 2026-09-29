# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=mips64el-unknown-linux -target-abi n32 -filetype=obj -o %t/test_ELF_N32.o %s
# RUN: llc -mtriple=mips64el-unknown-linux -target-abi n32 -filetype=obj -o %t/test_ELF_ExternalFunction_N32.o %S/Inputs/ExternalFunction.ll
# RUN: llvm-rtdyld -triple=mips64el-unknown-linux -verify -map-section test_ELF_N32.o,.text=0x1000 -map-section test_ELF_ExternalFunction_N32.o,.text=0x10000 -check=%s %t/test_ELF_N32.o %t/test_ELF_ExternalFunction_N32.o

# RUN: llvm-mc -triple=mips64-unknown-linux -target-abi n32 -filetype=obj -o %t/test_ELF_N32.o %s
# RUN: llc -mtriple=mips64-unknown-linux -target-abi n32 -filetype=obj -o %t/test_ELF_ExternalFunction_N32.o %S/Inputs/ExternalFunction.ll
# RUN: llvm-rtdyld -triple=mips64-unknown-linux -verify -map-section test_ELF_N32.o,.text=0x1000 -map-section test_ELF_ExternalFunction_N32.o,.text=0x10000 -check=%s %t/test_ELF_N32.o %t/test_ELF_ExternalFunction_N32.o

	.data
# Test R_MIPS_PC32 relocation.
# rtdyld-check: *{4}(R_MIPS_PC32) = (foo - R_MIPS_PC32)[31:0]
R_MIPS_PC32:
	.word foo-.
# rtdyld-check: *{4}(R_MIPS_PC32 + 4) = (foo - tmp1)[31:0]
tmp1:
	.4byte foo-tmp1

	.text
	.abicalls
	.section	.mdebug.abi64,"",@progbits
	.nan	legacy
	.file	"ELF_N32_PIC_relocations.ll"
	.text
	.globl	bar
	.align	3
	.type	bar,@function
	.set	nomicromips
	.set	nomips16
	.ent	bar
bar:
	.frame	$fp,40,$ra
	.mask 	0x00000000,0
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
	.set	noat
	daddiu	$sp, $sp, -40
	sd	$ra, 32($sp)
	sd	$fp, 24($sp)
	move	 $fp, $sp
	sd	$4, 16($fp)
	lb	$2, 0($4)
	sd	$4, 8($fp)

# Test R_MIPS_26 relocation.
# rtdyld-check:  decode_operand(insn1, 0)[27:0] = stub_addr(test_ELF_N32.o/.text, foo)[27:0]
insn1:
	.option pic0
	jal   foo
	.option pic2
	nop

# Test R_MIPS_PC16 relocation.
# rtdyld-check:  decode_operand(insn2, 1)[17:0] = (foo - insn2)[17:0]
insn2:
	bal   foo
	nop

	move	 $sp, $fp
	ld	$ra, 32($sp)
	ld	$fp, 24($sp)
	daddiu	$sp, $sp, 32
	jr	$ra
	nop
	.set	at
	.set	macro
	.set	reorder
	.end	bar
$func_end0:
	.size	bar, ($func_end0)-bar

	.globl	main
	.align	3
	.type	main,@function
	.set	nomicromips
	.set	nomips16
	.ent	main
main:
	.frame	$fp,32,$ra
	.mask 	0x00000000,0
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
	.set	noat
	daddiu	$sp, $sp, -32
	sd	$ra, 24($sp)
	sd	$fp, 16($sp)
	sd	$gp, 8($sp)
	move	 $fp, $sp

# $gp register contains address of the .got section + 0x7FF0. 0x7FF0 is
# the offset of $gp from the beginning of the .got section. Check that we are
# loading address of the page pointer from correct offset. In this case
# the page pointer is the first entry in the .got section, so offset will be
# 0 - 0x7FF0.
# rtdyld-check:  decode_operand(insn5, 2)[15:0] = 0x8010
#
# Check that the global offset table contains the page pointer.
# rtdyld-check: *{4}(section_addr(test_ELF_N32.o, .got)) = (_str + 0x8000) & 0xffff0000
insn5:
	ld	$25, %got_page(_str)($1)

# Check the offset of _str from the page pointer.
# rtdyld-check:  decode_operand(insn6, 2)[15:0] = _str[15:0]
insn6:
	daddiu	$25, $25, %got_ofst(_str)

# Check that we are loading address of var from correct offset. In this case
# var is the second entry in the .got section, so offset will be 4 - 0x7FF0.
# rtdyld-check:  decode_operand(insn7, 2)[15:0] = 0x8014
#
# Check that the global offset table contains the address of the var.
# rtdyld-check: *{4}(section_addr(test_ELF_N32.o, .got) + 4) = var
insn7:
	ld	$2, %got_disp(var)($1)
	sd	$25, 0($2)

# Check that we are loading address of bar from correct offset. In this case
# bar is the third entry in the .got section, so offset will be 8 - 0x7FF0.
# rtdyld-check:  decode_operand(insn8, 2)[15:0] = 0x8018
#
# Check that the global offset table contains the address of the bar.
# rtdyld-check: *{4}(section_addr(test_ELF_N32.o, .got) + 8) = bar
insn8:
	ld	$2, %call16(bar)($1)

	move	 $4, $25
	move	 $gp, $1
	move	 $25, $2
	jalr	$25
	nop
	move	 $sp, $fp
	ld	$gp, 8($sp)
	ld	$fp, 16($sp)
	ld	$ra, 24($sp)
	daddiu	$sp, $sp, 32
	jr	$ra
	nop
	.set	at
	.set	macro
	.set	reorder
	.end	main
$func_end1:
	.size	main, ($func_end1)-main

	.type	_str,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
_str:
	.asciz	"test"
	.size	_str, 5

	.type	var,@object
	.comm	var,8,8

	.section	".note.GNU-stack","",@progbits
	.text
