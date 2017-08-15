# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=armv7s-apple-ios7.0.0 -filetype=obj -o %t/foo.o %s
# RUN: llvm-rtdyld -triple=armv7s-apple-ios7.0.0 -verify -check=%s %t/foo.o

        .syntax unified
        .section        __TEXT,__text,regular,pure_instructions
        .globl  bar
        .align  2
bar:
# Check lower 16-bits of section difference relocation
# rtdyld-check: decode_operand(insn1, 1) = (foo$non_lazy_ptr-(nextPC+8))[15:0]
insn1:
        movw    r0, :lower16:(foo$non_lazy_ptr-(nextPC+8))
# Check upper 16-bits of section difference relocation
# rtdyld-check: decode_operand(insn2, 2) = (foo$non_lazy_ptr-(nextPC+8))[31:16]
insn2:
        movt    r0, :upper16:(foo$non_lazy_ptr-(nextPC+8))
nextPC:
        add     r1, r0, r0

# Check stub generation for external symbols by referencing a common symbol, 'baz'.
# Check both the content of the stub, and the reference to the stub.
# Stub should contain '0xe51ff004' (ldr pc, [pc, #-4]), followed by the target.
#
# rtdyld-check: *{4}(stub_addr(foo.o, __text, baz)) = 0xe51ff004
# rtdyld-check: *{4}(stub_addr(foo.o, __text, baz) + 4) = baz
#
# rtdyld-check: decode_operand(insn3, 0) = stub_addr(foo.o, __text, baz) - (insn3 + 8)
insn3:
        bl      baz

# Check stub generation for internal symbols by referencing 'bar'.
# rtdyld-check: *{4}(stub_addr(foo.o, __text, bar) + 4) = bar
insn4:
        bl      bar
        bx	lr

# Add 'aaa' to the common symbols to make sure 'baz' isn't at the start of the
# section. This ensures that we test VANILLA relocation addends correctly.
        .comm   aaa, 4, 2
        .comm   baz, 4, 2
        .comm   foo, 4, 2

	.section        __DATA,__data
	.globl  _a
	.align  2
# rtdyld-check: *{4}bar_ofs = bar + 4
bar_ofs:
	.long   bar + 4

# Check that the symbol pointer section entries are fixed up properly:
# rtdyld-check: *{4}foo$non_lazy_ptr = foo
        .section	__DATA,__nl_symbol_ptr,non_lazy_symbol_pointers
  	.align	2
foo$non_lazy_ptr:
	.indirect_symbol	foo
	.long	0

.subsections_via_symbols
