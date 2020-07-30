# RUN: llvm-mc -triple=powerpc64le-unknown-unknown -filetype=obj %s 2>&1 | \
# RUN: FileCheck %s -check-prefix=MC
# RUN: llvm-mc -triple=powerpc64le-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -r | FileCheck %s -check-prefix=READOBJ

# RUN: llvm-mc -triple=powerpc64-unknown-unknown -filetype=obj %s 2>&1 | \
# RUN: FileCheck %s -check-prefix=MC
# RUN: llvm-mc -triple=powerpc64-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -r | FileCheck %s -check-prefix=READOBJ

# This test checks that on Power PC we can correctly convert @pcrel and
# @got@pcrel into R_PPC64_PCREL34 and R_PPC64_GOT_PCREL34.

# MC-NOT:    error: invalid variant

# READOBJ:        0x0 R_PPC64_PCREL34 locvalue 0x0
# READOBJ-NEXT:   0x20 R_PPC64_GOT_PCREL34 evalue 0x0

	.text
	.abiversion 2
	.globl	local                   # -- Begin function local
	.p2align	4
	.type	local,@function
local:                                  # @local
.Llocal$local:
.Lfunc_begin0:
# %bb.0:                                # %entry
	plwa 3, locvalue@PCREL(0), 1
	blr
	.long	0
	.quad	0
.Lfunc_end0:
	.size	local, .Lfunc_end0-.Lfunc_begin0
                                        # -- End function
	.globl	external                # -- Begin function external
	.p2align	4
	.type	external,@function
external:                               # @external
.Lexternal$local:
.Lfunc_begin1:
# %bb.0:                                # %entry
	pld 3, evalue@got@pcrel(0), 1
	lwa 3, 0(3)
	blr
	.long	0
	.quad	0
.Lfunc_end1:
	.size	external, .Lfunc_end1-.Lfunc_begin1
                                        # -- End function
	.type	locvalue,@object        # @locvalue
	.section	.bss,"aw",@nobits
	.globl	locvalue
	.p2align	2
locvalue:
.Llocvalue$local:
	.long	0                       # 0x0
	.size	locvalue, 4

