# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=bpfel -filetype=obj -o %t/test_reloc_abs64.o %s
# RUN: llvm-rtdyld -triple=bpfel -verify -check=%s %t/test_reloc_abs64.o

# test R_BPF_64_ABS64 which should have relocation resolved properly.

	.text
	.file	"t1.c"
	.globl	g                               # -- Begin function g
	.p2align	3
	.type	g,@function
g:                                      # @g
	r0 = 0
	exit
.Lfunc_end0:
	.size	g, .Lfunc_end0-g
                                        # -- End function
	.type	gbl,@object                     # @gbl
	.data
	.globl	gbl
	.p2align	3
gbl:
	.quad	g
	.size	gbl, 8

# rtdyld-check: *{8}gbl = section_addr(test_reloc_abs64.o, .text)
