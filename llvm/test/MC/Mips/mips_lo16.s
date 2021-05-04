# PR49821: Check that R_MIPS_LO16 relocs do not wrap around with large addends.

# RUN: llvm-mc %s -triple mips-unknown-unknown -filetype=obj | \
# RUN:	 llvm-objdump -d -r --no-show-raw-insn - | \
# RUN:   FileCheck -check-prefix=MIPS32 %s

# RUN: llvm-mc %s -triple mips64-unknown-unknown -filetype=obj | \
# RUN:	 llvm-objdump -d -r --no-show-raw-insn - | \
# RUN:   FileCheck -check-prefix=MIPS64 %s

	.text
foo:
	lui	$2, %hi(bar)
# MIPS32: 00000000:  R_MIPS_HI16  bar
# MIPS64: 0000000000000000:  R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE	.rodata.str1.1+0x8000
	addiu	$2, $2, %lo(bar)
# MIPS32: 00000004:  R_MIPS_LO16  bar
# MIPS64: 0000000000000004:  R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE	.rodata.str1.1+0x8000
	.section	.rodata.str1.1,"aMS",@progbits,1
	.zero 0x8000
bar:
	.asciz	"hello"
