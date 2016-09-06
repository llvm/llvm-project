# RUN: llvm-mc -triple=mips64el-unknown-linux -mcpu=mips64r6 -code-model=small -filetype=obj -o %T/test_ELF_N64R6.o %s
# RUN: llc -mtriple=mips64el-unknown-linux -mcpu=mips64r6 -filetype=obj -o %T/test_ELF_ExternalFunction_N64R6.o %S/Inputs/ExternalFunction.ll
# RUN: llvm-rtdyld -triple=mips64el-unknown-linux -mcpu=mips64r6 -verify -map-section test_ELF_N64R6.o,.text=0x1000 -map-section test_ELF_ExternalFunction_N64R6.o,.text=0x10000 -check=%s %/T/test_ELF_N64R6.o %T/test_ELF_ExternalFunction_N64R6.o

# RUN: llvm-mc -triple=mips64-unknown-linux -mcpu=mips64r6 -code-model=small -filetype=obj -o %T/test_ELF_N64R6.o %s
# RUN: llc -mtriple=mips64-unknown-linux -mcpu=mips64r6 -filetype=obj -o %T/test_ELF_ExternalFunction_N64R6.o %S/Inputs/ExternalFunction.ll
# RUN: llvm-rtdyld -triple=mips64-unknown-linux -mcpu=mips64r6 -verify -map-section test_ELF_N64R6.o,.text=0x1000 -map-section test_ELF_ExternalFunction_N64R6.o,.text=0x10000 -check=%s %/T/test_ELF_N64R6.o %T/test_ELF_ExternalFunction_N64R6.o

	.text
	.abicalls
	.nan	2008
	.text
	.set	nomicromips
	.set	nomips16
	.set	noreorder
	.set	nomacro
	.set	noat

	.align	3
	.globl	bar
	.type	bar,@function

bar:
# Test R_MIPS_PC18_S3 relocation.
# rtdyld-check:  decode_operand(R_MIPS_PC18_S3, 1)[20:0] = (foo - R_MIPS_PC18_S3)[20:0]
R_MIPS_PC18_S3:
	ldpc $6,foo

# Test R_MIPS_PC19_S2 relocation.
# rtdyld-check:  decode_operand(R_MIPS_PC19_S2, 1)[20:0] = (foo - R_MIPS_PC19_S2)[20:0]
R_MIPS_PC19_S2:
	lwpc $6,foo

# Test R_MIPS_PC21_S2 relocation.
# rtdyld-check:  decode_operand(R_MIPS_PC21_S2, 1)[22:0] = (foo - next_pc(R_MIPS_PC21_S2) + 0x04)[22:0]
R_MIPS_PC21_S2:
	bnezc	$5,foo

# Test R_MIPS_PC26_S2 relocation.
# rtdyld-check:  decode_operand(R_MIPS_PC26_S2, 0)[27:0] = (foo - next_pc(R_MIPS_PC26_S2) + 0x04)[27:0]
R_MIPS_PC26_S2:
	balc	foo

# Test R_MIPS_PCHI16 relocation.
# rtdyld-check:  decode_operand(R_MIPS_PCHI16, 1)[15:0] = (foo - R_MIPS_PCHI16 + 0x8000)[31:16]
R_MIPS_PCHI16:
	aluipc $5, %pcrel_hi(foo)

# Test R_MIPS_PCLO16 relocation.
# rtdyld-check:  decode_operand(R_MIPS_PCLO16, 2)[15:0] = (foo - R_MIPS_PCLO16)[15:0]
R_MIPS_PCLO16:
	addiu  $5, $5, %pcrel_lo(foo)

	.size	bar, .-bar
