# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj -o %t/ppc64_elf.o %s
# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj -o %t/ppc64_elf_module_b.o %S/Inputs/ppc64_elf_module_b.s
# RUN: llvm-rtdyld -triple=powerpc64le-unknown-linux-gnu -verify -check=%s %t/ppc64_elf.o %t/ppc64_elf_module_b.o

       	.text
	.abiversion 2
	.file	"Module2.ll"
	.globl	bar                     # -- Begin function bar
	.p2align	4
	.type	bar,@function
.Lfunc_toc0:                            # @bar
	.quad	.TOC.-.Lfunc_gep0
bar:
.Lfunc_begin0:
	.cfi_startproc
.Lfunc_gep0:
	ld 2, .Lfunc_toc0-.Lfunc_gep0(12)
	add 2, 2, 12
.Lfunc_lep0:
	.localentry	bar, .Lfunc_lep0-.Lfunc_gep0
# %bb.0:
	mflr 0
	std 0, 16(1)
	stdu 1, -32(1)
	.cfi_def_cfa_offset 32
	.cfi_offset lr, 16
# rtdyld-check: (*{4}(stub_addr(ppc64_elf.o, .text, foo) +  0)) [15:0] = foo_gep [63:48]
# rtdyld-check: (*{4}(stub_addr(ppc64_elf.o, .text, foo) +  4)) [15:0] = foo_gep [47:32]
# rtdyld-check: (*{4}(stub_addr(ppc64_elf.o, .text, foo) + 12)) [15:0] = foo_gep [31:16]
# rtdyld-check: (*{4}(stub_addr(ppc64_elf.o, .text, foo) + 16)) [15:0] = foo_gep [16:0]
# rtdyld-check: decode_operand(foo_call, 0) = (stub_addr(ppc64_elf.o, .text, foo) - foo_call) >> 2
foo_call:
	bl foo
	nop
	addi 1, 1, 32
	ld 0, 16(1)
	mtlr 0
	blr
	.long	0
	.quad	0
.Lfunc_end0:
	.size	bar, .Lfunc_end0-.Lfunc_begin0
	.cfi_endproc
                                        # -- End function

	.section	".note.GNU-stack","",@progbits
