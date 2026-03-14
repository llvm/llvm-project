# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj -o %t/test_ELF1_x86-64.o %s
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj -o %t/test_ELF2_x86-64.o %s
# RUN: llc -mtriple=x86_64-pc-linux -filetype=obj -o %t/test_ELF_ExternalGlobal_x86-64.o %S/Inputs/ExternalGlobal.ll
# RUN: llvm-rtdyld -triple=x86_64-pc-linux -verify %t/test_ELF1_x86-64.o  %t/test_ELF_ExternalGlobal_x86-64.o
# Test that we can load this code twice at memory locations more than 2GB apart
# RUN: llvm-rtdyld -triple=x86_64-pc-linux -verify -map-section test_ELF1_x86-64.o,.got=0x10000 -map-section test_ELF2_x86-64.o,.text=0x100000000 -map-section test_ELF2_x86-64.o,.got=0x100010000 %t/test_ELF1_x86-64.o %t/test_ELF2_x86-64.o %t/test_ELF_ExternalGlobal_x86-64.o

# Assembly obtained by compiling the following and adding checks:
# @G = external global i8*
#
# define i8* @foo() {
#    %ret = load i8** @G
#    ret i32 %ret
# }
#

#
	.text
	.file	"ELF_x64-64_PIC_relocations.ll"
	.align	16, 0x90
	.type	foo,@function
foo:                                    # @foo
# %bb.0:
	movq	G@GOTPCREL(%rip), %rax
	movl	(%rax), %eax
	retq
.Ltmp0:
	.size	foo, .Ltmp0-foo


	.section	".note.GNU-stack","",@progbits
