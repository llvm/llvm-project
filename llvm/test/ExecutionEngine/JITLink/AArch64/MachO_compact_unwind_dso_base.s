# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=arm64-apple-darwin -filetype=obj -o %t/pos.o %s
# RUN: llvm-mc -triple=arm64-apple-darwin --defsym DSO_BASE_ABOVE=1 \
# RUN:     -filetype=obj -o %t/neg.o %s
#
# Check that the __unwind_info writer chooses a compact-unwind base that is at
# or below every covered address, so that the unsigned 32-bit deltas it encodes
# never underflow -- even when the object's code is emitted at a high address.
#
# Positive: with no "__jitlink$libunwind_dso_base" symbol the writer synthesizes
# a per-graph local Mach-O header (laid out first, so lowest), and linking the
# object at a high slab address succeeds.
# RUN: llvm-jitlink -noexec -num-threads=0 -entry=_main \
# RUN:     -slab-allocate 1Mb -slab-address 0x800000000000 -slab-page-size 4096 \
# RUN:     %t/pos.o
#
# Negative: pinning the base *above* the code (as a shared JITDylib header does
# when a graph lands below it) underflows and aborts linking. This is the bug
# that a per-graph base avoids.
# RUN: not llvm-jitlink -noexec -num-threads=0 -entry=_main \
# RUN:     -slab-allocate 1Mb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     %t/neg.o 2>&1 | FileCheck %s
#
# CHECK: exceeds 32 bits

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	2
_main:
	.cfi_startproc
	ret
	.cfi_endproc

.ifdef DSO_BASE_ABOVE
	.globl	__jitlink$libunwind_dso_base
	.set	__jitlink$libunwind_dso_base, 0xffff000000000000
.endif

.subsections_via_symbols
