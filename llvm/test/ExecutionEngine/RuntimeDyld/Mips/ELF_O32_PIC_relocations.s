# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=mipsel-unknown-linux -filetype=obj -o %t/test_ELF_O32.o %s
# RUN: llc -mtriple=mipsel-unknown-linux -filetype=obj -o %t/test_ELF_ExternalFunction_O32.o %S/Inputs/ExternalFunction.ll
# RUN: llvm-rtdyld -triple=mipsel-unknown-linux -verify -map-section test_ELF_O32.o,"<common symbols>"=0x7FF8 -map-section test_ELF_O32.o,.text=0x1000 -map-section test_ELF_ExternalFunction_O32.o,.text=0x10000 -check=%s %t/test_ELF_O32.o %t/test_ELF_ExternalFunction_O32.o

# RUN: llvm-mc -triple=mips-unknown-linux -filetype=obj -o %t/test_ELF_O32.o %s
# RUN: llc -mtriple=mips-unknown-linux -filetype=obj -o %t/test_ELF_ExternalFunction_O32.o %S/Inputs/ExternalFunction.ll
# RUN: llvm-rtdyld -triple=mips-unknown-linux -verify -map-section test_ELF_O32.o,"<common symbols>"=0x7FF8 -map-section test_ELF_O32.o,.text=0x1000 -map-section test_ELF_ExternalFunction_O32.o,.text=0x10000 -check=%s %t/test_ELF_O32.o %t/test_ELF_ExternalFunction_O32.o

        .data
# rtdyld-check: *{4}R_MIPS_32 = foo[31:0]
R_MIPS_32:
        .word foo
# rtdyld-check: *{4}(R_MIPS_32+4) = foo[31:0]
        .4byte foo
# rtdyld-check: *{4}(R_MIPS_PC32) = (foo - R_MIPS_PC32)[31:0]
R_MIPS_PC32:
        .word foo-.
# rtdyld-check: *{4}(R_MIPS_PC32 + 4) = (foo - tmp1)[31:0]
tmp1:
        .4byte foo-tmp1

	.text
	.abicalls
	.nan	legacy
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
# rtdyld-check:  decode_operand(R_MIPS_26, 0)[27:0] = stub_addr(test_ELF_O32.o, .text, foo)[27:0]
# rtdyld-check:  decode_operand(R_MIPS_26, 0)[1:0] = 0
R_MIPS_26:
	j   foo
	nop

# rtdyld-check:  decode_operand(R_MIPS_PC16, 1)[17:0] = (foo - R_MIPS_PC16)[17:0]
R_MIPS_PC16:
	bal   foo
	nop

# rtdyld-check:  decode_operand(R_MIPS_HI16, 1)[15:0] = foo[31:16]
R_MIPS_HI16:
	lui	$1, %hi(foo)

# rtdyld-check:  decode_operand(R_MIPS_LO16, 1)[15:0] = foo[15:0]
R_MIPS_LO16:
	lui	$1, %lo(foo)

# rtdyld-check:  decode_operand(R_MIPS_HI16_ADDEND, 1)[15:0] = (var+0x8008)[31:16]
R_MIPS_HI16_ADDEND:
	lui	$2, %hi(var+8)

# rtdyld-check:  decode_operand(R_MIPS_LO16_ADDEND, 2)[15:0] = (var+0x8)[15:0]
R_MIPS_LO16_ADDEND:
	lb	$2, %lo(var+8)($2)

	.size	bar, .-bar
	.comm	var,9,1
