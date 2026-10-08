# Verify the COFF __imp_ IAT synthesis pass: for a dllimport reference to an
# undefined __imp_X symbol, JITLink should define __imp_X over an 8-byte pointer
# slot that holds the address of X (resolved as an ordinary external). Both the
# call form (callq *__imp_X) and the data-access form (movq __imp_X) resolve
# indirectly through that slot.
#
# X (foo/bar) is supplied as an absolute symbol, so no real library is needed --
# this exercises the pass itself, not any resolution mechanism.
#
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t.o
# RUN: llvm-jitlink -noexec \
# RUN:              -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN:              -abs foo=0x7ff700000000 \
# RUN:              -abs bar=0x7ff700001000 \
# RUN:              -check %s %t.o

	.text

	.def main;
	.scl 2;
	.type 32;
	.endef
	.globl main
	.p2align 4, 0x90
main:
	retq

# The synthesized __imp_bar slot holds bar's address...
# jitlink-check: *{8}(__imp_bar) = bar
# ... and the dllimport call reads through that slot (RIP-relative displacement
# of the indirect call's memory operand, MCInst operand 3).
# jitlink-check: decode_operand(test_call, 3) = __imp_bar - next_pc(test_call)
	.def test_call;
	.scl 2;
	.type 32;
	.endef
	.globl test_call
	.p2align 4, 0x90
test_call:
	callq *__imp_bar(%rip)
	retq

# Same for a data access: the __imp_foo slot holds foo's address, and the load
# reads through it (displacement is MCInst operand 4 for `movq mem, reg`).
# jitlink-check: *{8}(__imp_foo) = foo
# jitlink-check: decode_operand(test_load, 4) = __imp_foo - next_pc(test_load)
	.def test_load;
	.scl 2;
	.type 32;
	.endef
	.globl test_load
	.p2align 4, 0x90
test_load:
	movq __imp_foo(%rip), %rax
	retq
